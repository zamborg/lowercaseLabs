#!/usr/bin/env python3
"""Generate the Controllr Voice shortcut from the app's action manifests."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import plistlib
import uuid

HERE = Path(__file__).resolve().parent
MANIFESTS = HERE.parent / "ControllrCore/Sources/ControllrCore/Actions"
NAME = "Controllr Voice 1.6"
BUNDLE = "com.lowercaselabs.controllr"


@dataclass(frozen=True)
class Reference:
    value: dict

    @classmethod
    def variable(cls, name):
        return cls({"Type": "Variable", "VariableName": name})

    def selecting_properties(self, *names):
        return Reference({**self.value, "Aggrandizements": [
            {"Type": "WFPropertyVariableAggrandizement", "PropertyName": name} for name in names
        ]})

    @property
    def attachment(self):
        return {"Value": self.value, "WFSerializationType": "WFTextTokenAttachment"}

    @property
    def text(self):
        return {
            "Value": {"string": "\ufffc", "attachmentsByRange": {"{0, 1}": self.value}},
            "WFSerializationType": "WFTextTokenString",
        }


class Workflow:
    def __init__(self, team_id):
        self.actions = []
        self.team_id = team_id
        self.next_id = 0

    def identifier(self):
        self.next_id += 1
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{BUNDLE}/voice/{self.next_id}")).upper()

    def add(self, action, **params):
        self.actions.append({"WFWorkflowActionIdentifier": action, "WFWorkflowActionParameters": params})

    def system(self, name, **params):
        self.add(f"is.workflow.actions.{name}", **params)

    def output(self, action, name, **params):
        identifier = self.identifier()
        self.add(action, UUID=identifier, CustomOutputName=name, **params)
        return Reference({"Type": "ActionOutput", "OutputUUID": identifier, "OutputName": name})

    def text(self, value, name):
        return self.output("is.workflow.actions.gettext", name, WFTextActionText=value)

    def key(self, dictionary, key, *, as_text=True):
        value = self.output("is.workflow.actions.getvalueforkey", key,
                            WFInput=dictionary.attachment, WFDictionaryKey=key)
        return self.text(value.text, key.title()) if as_text else value

    def set_variable(self, name, ref):
        self.system("setvariable", WFVariableName=name, WFInput=ref.attachment)

    def comment(self, text):
        self.system("comment", WFCommentActionText=text)

    def app_intent(self, intent, result_name=None, **params):
        descriptor = {"AppIntentIdentifier": intent, "BundleIdentifier": BUNDLE,
                      "Name": "controllr", "TeamIdentifier": self.team_id}
        params["AppIntentDescriptor"] = descriptor
        action = f"{BUNDLE}.{intent}"
        if result_name:
            return self.output(action, result_name, **params)
        self.add(action, **params)

    @contextmanager
    def when(self, ref, value):
        label = "the requested value" if isinstance(value, Reference) else value
        self.comment(f"- Run this branch when the input is {label}.")
        group = self.identifier()
        self.system("conditional", GroupingIdentifier=group, WFControlFlowMode=0,
                    WFInput={"Type": "Variable", "Variable": ref.attachment},
                    WFCondition=4, WFConditionalActionString=value.text if isinstance(value, Reference) else value)

        def otherwise():
            self.system("conditional", GroupingIdentifier=group, WFControlFlowMode=1)

        yield otherwise
        self.system("conditional", GroupingIdentifier=group, WFControlFlowMode=2)

    @contextmanager
    def repeat(self, count):
        self.comment("- Listen when the thread expects a reply.\n- Route action results before listening again.")
        group = self.identifier()
        self.system("repeat.count", GroupingIdentifier=group, WFControlFlowMode=0, WFRepeatCount=count)
        yield
        self.system("repeat.count", GroupingIdentifier=group, WFControlFlowMode=2)

    @contextmanager
    def repeat_each(self, items):
        self.comment("- Read the next recent message.\n- Check its sender or conversation before returning its body.")
        group = self.identifier()
        self.system("repeat.each", GroupingIdentifier=group, WFControlFlowMode=0, WFInput=items.attachment)
        yield Reference.variable("Repeat Item 2")
        self.system("repeat.each", GroupingIdentifier=group, WFControlFlowMode=2, UUID=self.identifier())

    def speak(self, ref):
        self.system("speaktext", WFText=ref.text, WFSpeakTextWait=True)

    def confirm(self, summary, execute, cancel):
        self.comment("- Show the proposed action.\n- Run it only when the user selects Continue.")
        group = self.identifier()
        self.system("choosefrommenu", GroupingIdentifier=group, WFControlFlowMode=0,
                    WFMenuPrompt=summary.text, WFMenuItems=["Continue", "Cancel"])
        self.system("choosefrommenu", GroupingIdentifier=group, WFControlFlowMode=1, WFMenuItemTitle="Continue")
        execute()
        self.system("choosefrommenu", GroupingIdentifier=group, WFControlFlowMode=1, WFMenuItemTitle="Cancel")
        cancel()
        self.system("choosefrommenu", GroupingIdentifier=group, WFControlFlowMode=2)


def send_message(workflow, payload, manifest):
    body = workflow.key(payload, "body")
    recipient = workflow.key(payload, "recipient")
    workflow.system("sendmessage", WFSendMessageContent=body.text,
                    WFSendMessageActionRecipients=recipient.text,
                    IntentAppDefinition={"BundleIdentifier": "com.apple.MobileSMS", "Name": "Messages",
                                         "TeamIdentifier": "0000000000"})


def media(workflow, payload, manifest):
    command = workflow.key(payload, "command")
    for verb in ("play", "pause"):
        with workflow.when(command, verb):
            workflow.system("pausemusic", WFPlayPauseBehavior=verb.title())
    for verb, action in (("next", "skipforward"), ("previous", "skipback")):
        with workflow.when(command, verb):
            workflow.system(action)


def timer(workflow, payload, manifest):
    duration = workflow.key(payload, "seconds_total")
    for seconds in manifest["supportedSeconds"]:
        with workflow.when(duration, str(seconds)):
            unit, magnitude = ("min", seconds // 60) if seconds % 60 == 0 else ("sec", seconds)
            workflow.system("timer.start", WFDuration={"Value": {"Unit": unit, "Magnitude": magnitude},
                                                       "WFSerializationType": "WFQuantityFieldValue"})


def focus(workflow, payload, manifest):
    mode = workflow.key(payload, "focus")
    modes = {
        "dnd": {"Enabled": True},
        "sleep": {"Enabled": True, "FocusModes": {"DisplayString": "Sleep", "Identifier": "com.apple.sleep.sleep-mode"}},
        "work": {"Enabled": True, "FocusModes": {"DisplayString": "Work", "Identifier": "com.apple.focus.work"}},
        "off": {"Enabled": False},
    }
    for label, params in modes.items():
        with workflow.when(mode, label):
            workflow.system("dnd.set", **params)


def flashlight(workflow, payload, manifest):
    setting = workflow.key(payload, "setting")
    for state in ("Off", "On"):
        with workflow.when(setting, state):
            workflow.system("flashlight", WFFlashlightSetting=state)


def alarm(workflow, payload, manifest):
    time = workflow.key(payload, "time")
    label = workflow.key(payload, "label")
    workflow.add("com.apple.mobiletimer-framework.MobileTimerIntents.MTCreateAlarmIntent",
                 AppIntentDescriptor={"AppIntentIdentifier": "CreateAlarmIntent", "BundleIdentifier": "com.apple.mobiletimer",
                                      "Name": "Clock", "TeamIdentifier": "0000000000"},
                 dateComponents=time.text, name=label.text)

def read_clipboard(workflow, payload, manifest):
    return workflow.output("is.workflow.actions.getclipboard", "Clipboard")


def recent_messages(workflow, limit, **filters):
    return workflow.output(
        "com.apple.MobileSMS.MessageEntity", "Messages",
        AppIntentDescriptor={"AppIntentIdentifier": "MessageEntity", "BundleIdentifier": "com.apple.MobileSMS",
                             "Name": "Messages", "TeamIdentifier": "0000000000"},
        WFCompoundType=1, WFContentItemSortProperty="date", WFContentItemSortOrder="Latest First",
        WFContentItemLimitEnabled=True, WFContentItemLimitNumber=limit, **filters,
    )


def find_messages(workflow, payload, manifest):
    phrase = workflow.key(payload, "query")
    message = recent_messages(workflow, 1, WFContentItemFilter={
        "WFSerializationType": "WFContentPredicateTableTemplate",
        "Value": {"WFActionParameterFilterPrefix": 1, "WFContentPredicateBoundedDate": False,
                  "WFActionParameterFilterTemplates": [
                      {"Property": "body", "Operator": 99, "Removable": True, "Values": {"String": phrase.text}}
                  ]},
    })
    return message.selecting_properties("body")


def find_recent_messages(workflow, payload, manifest):
    scope = workflow.key(payload, "scope")
    name = workflow.key(payload, "name")
    results = Reference.variable("Matched Messages")
    found = Reference.variable("Found Message")
    workflow.set_variable("Matched Messages", workflow.text("No match", "No Matches"))
    workflow.set_variable("Found Message", workflow.text("no", "Searching"))

    def capture(label, message):
        with workflow.when(label, name):
            body = message.selecting_properties("body")
            text = {"WFSerializationType": "WFTextTokenString",
                    "Value": {"string": "\ufffc\n\ufffc", "attachmentsByRange": {"{0, 1}": label.value, "{2, 1}": body.value}}}
            workflow.set_variable("Matched Messages", workflow.text(text, "Matched Message"))
            workflow.set_variable("Found Message", workflow.text("yes", "Found"))

    with workflow.repeat_each(recent_messages(workflow, 120)) as message:
        with workflow.when(found, "no"):
            with workflow.when(scope, "sender") as conversation:
                sender = workflow.text(message.selecting_properties("author", "person").text, "Sender Name")
                capture(sender, message)
                conversation()
                group = workflow.text(message.selecting_properties("conversation", "displayName").text, "Conversation Name")
                capture(group, message)
    return results


def set_brightness(workflow, payload, manifest):
    level = workflow.key(payload, "level", as_text=False)
    workflow.system("setbrightness", WFBrightness=level.attachment)


def set_volume(workflow, payload, manifest):
    level = workflow.key(payload, "level", as_text=False)
    workflow.system("setvolume", WFVolume=level.attachment)


HANDLERS = {
    "messages": send_message,
    "media": media,
    "timer": timer,
    "focus": focus,
    "flashlight": flashlight,
    "alarm": alarm,
    "read_clipboard": read_clipboard,
    "find_messages": find_messages,
    "find_recent_messages": find_recent_messages,
    "set_brightness": set_brightness,
    "set_volume": set_volume,
}


def open_url(workflow, payload, manifest):
    url = workflow.key(payload, "url")
    workflow.system("openurl", WFInput=url.attachment)


def dispatch(workflow, decision, summary, thread_id):
    module = workflow.key(decision, "shortcut")
    action_id = workflow.key(decision, "action_id")
    payload = workflow.key(decision, "input", as_text=False)
    confirmation = workflow.key(decision, "confirmation")
    callback = {"threadID": thread_id.text, "actionID": action_id.text}

    def cancel():
        workflow.app_intent("RecordActionIntent", **callback, completed=False)
        workflow.system("exit")

    def approve():
        workflow.set_variable("Approval", workflow.text("approved", "Approved"))

    workflow.set_variable("Approval", workflow.text("automatic", "Automatic"))
    with workflow.when(confirmation, "ask"):
        workflow.confirm(summary, approve, cancel)

    authorization = workflow.app_intent("AuthorizeActionIntent", result_name="Authorization", **callback,
                                        approval=Reference.variable("Approval").text)

    def branch(identifier, handler=None, manifest=None):
        with workflow.when(module, identifier):
            output = handler(workflow, payload, manifest) if handler else None
            if output:
                workflow.set_variable("Action Output", output)
            workflow.set_variable("Action Outcome", read if output else write)

    with workflow.when(authorization, "ready") as denied:
        workflow.set_variable("Action Output", workflow.text("No results.", "Empty Read Output"))
        workflow.set_variable("Action Outcome", workflow.text("unavailable", "Unhandled"))
        outcome = Reference.variable("Action Outcome")
        read = workflow.text("read", "Read Completed")
        write = workflow.text("write", "Action Completed")
        branch("open_url", open_url)
        branch("open_url_confirm", open_url)
        branch("approved_native")
        for path in sorted(MANIFESTS.glob("*.json")):
            manifest = json.loads(path.read_text())
            if manifest["kind"] == "shortcut":
                branch(manifest["id"], HANDLERS[manifest["shortcutHandler"]], manifest)
        with workflow.when(outcome, "unavailable"):
            workflow.speak(workflow.text("This action isn't available in this shortcut.", "Unavailable Action"))
            workflow.system("exit")
        with workflow.when(outcome, "read") as regular_completion:
            response = workflow.app_intent("CompleteActionIntent", result_name="Read Decision", **callback,
                                           output=Reference.variable("Action Output").text)
            workflow.set_variable("Decision JSON", response)
            regular_completion()
            response = workflow.app_intent("CompleteActionIntent", result_name="Action Decision", **callback)
            workflow.set_variable("Decision JSON", response)
        workflow.set_variable("Next Step", workflow.text("dispatch", "Handle Result"))
        denied()
        workflow.speak(authorization)
        workflow.system("exit")


def voice_turn(workflow, current_thread):
    with workflow.when(Reference.variable("Next Step"), "dictate"):
        utterance = workflow.output("is.workflow.actions.dictatetext", "Dictated Text",
                                    WFDictateTextStopListening="After Pause")
        response = workflow.app_intent("DecideIntent", result_name="User Turn",
                                       transcript=utterance.text, threadID=current_thread.text, routerVersion=3)
        workflow.set_variable("Decision JSON", response)
    decision = workflow.output("is.workflow.actions.detect.dictionary", "Decision",
                               WFInput=Reference.variable("Decision JSON").attachment)
    thread_id = workflow.key(decision, "thread_id")
    workflow.set_variable("Thread ID", thread_id)
    summary = workflow.key(decision, "summary")
    response_type = workflow.key(decision, "action")
    voice_next = workflow.key(decision, "voice_next")

    with workflow.when(response_type, "run") as spoken:
        dispatch(workflow, decision, summary, thread_id)
        spoken()
        with workflow.when(voice_next, "listen") as finish:
            workflow.speak(summary)
            workflow.set_variable("Next Step", workflow.text("dictate", "Listen for Reply"))
            finish()
            with workflow.when(voice_next, "silent") as speak:
                workflow.system("exit")
                speak()
                workflow.speak(summary)
                workflow.system("exit")


def make_router(team_id):
    workflow = Workflow(team_id)
    workflow.comment("Dictate a request, continue the same Controllr thread, and run the proposed phone action.")
    workflow.comment("Shortcuts generated by Shortcuts Playground. Talk to Controllr by voice.")
    input_ref = Reference({"Type": "ExtensionInput"})
    thread = workflow.text(input_ref.text, "Input Thread ID")
    workflow.set_variable("Thread ID", thread)
    current_thread = Reference.variable("Thread ID")
    workflow.set_variable("Next Step", workflow.text("dictate", "Listen"))

    with workflow.repeat(32):
        voice_turn(workflow, current_thread)

    workflow.speak(workflow.text("Open Controllr to continue this thread.", "Continue in Controllr"))
    return {
        "WFWorkflowName": NAME,
        "WFWorkflowActions": workflow.actions,
        "WFWorkflowClientVersion": "2900",
        "WFWorkflowMinimumClientVersion": 900,
        "WFWorkflowIcon": {"WFWorkflowIconStartColor": 463140863, "WFWorkflowIconGlyphNumber": 59511},
        "WFWorkflowImportQuestions": [],
        "WFWorkflowInputContentItemClasses": ["WFStringContentItem"],
        "WFWorkflowTypes": [],
        "WFWorkflowHasOutputFallback": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-id", required=True, help="Apple Developer team used to sign the iOS app")
    parser.add_argument("--output", type=Path, default=HERE / "src" / f"{NAME}.xml")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(plistlib.dumps(make_router(args.team_id), fmt=plistlib.FMT_XML))
    print(args.output)


if __name__ == "__main__":
    main()
