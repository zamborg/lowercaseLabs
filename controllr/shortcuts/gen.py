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
NAME = "Controllr Voice"
BUNDLE = "com.lowercaselabs.controllr"


@dataclass(frozen=True)
class Reference:
    value: dict

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

    def app_intent(self, intent, output=None, **params):
        descriptor = {"AppIntentIdentifier": intent, "BundleIdentifier": BUNDLE,
                      "Name": "controllr", "TeamIdentifier": self.team_id}
        params["AppIntentDescriptor"] = descriptor
        action = f"{BUNDLE}.{intent}"
        if output:
            return self.output(action, output, **params)
        self.add(action, **params)

    @contextmanager
    def when(self, ref, value):
        self.comment(f"- Run this branch when the input is {value}.")
        group = self.identifier()
        self.system("conditional", GroupingIdentifier=group, WFControlFlowMode=0,
                    WFInput={"Type": "Variable", "Variable": ref.attachment},
                    WFCondition=4, WFConditionalActionString=value)

        def otherwise():
            self.system("conditional", GroupingIdentifier=group, WFControlFlowMode=1)

        yield otherwise
        self.system("conditional", GroupingIdentifier=group, WFControlFlowMode=2)

    @contextmanager
    def repeat(self, count):
        self.comment("- Dictate and pass the current thread ID.\n- Repeat only while Controllr asks for details.")
        group = self.identifier()
        self.system("repeat.count", GroupingIdentifier=group, WFControlFlowMode=0, WFRepeatCount=count)
        yield
        self.system("repeat.count", GroupingIdentifier=group, WFControlFlowMode=2)

    def speak(self, ref):
        self.system("speaktext", WFText=ref.text)

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


HANDLERS = {
    "messages": send_message,
    "media": media,
    "timer": timer,
    "focus": focus,
    "flashlight": flashlight,
    "alarm": alarm,
}


def open_url(workflow, payload, manifest):
    url = workflow.key(payload, "url")
    workflow.system("openurl", WFInput=url.attachment)


def dispatch(workflow, decision, summary, thread_id):
    module = workflow.key(decision, "shortcut")
    action_id = workflow.key(decision, "action_id")
    payload = workflow.key(decision, "input", as_text=False)

    def receipt(completed):
        workflow.app_intent("RecordActionIntent", threadID=thread_id.text, actionID=action_id.text, completed=completed)

    def branch(action_id, handler, manifest, confirm):
        with workflow.when(module, action_id):
            def execute():
                handler(workflow, payload, manifest)
                receipt(True)

            if confirm:
                workflow.confirm(summary, execute, lambda: receipt(False))
            else:
                execute()
            workflow.system("exit")

    branch("open_url", open_url, {}, False)
    branch("open_url_confirm", open_url, {}, True)
    for path in sorted(MANIFESTS.glob("*.json")):
        manifest = json.loads(path.read_text())
        if manifest["kind"] == "shortcut":
            handler = HANDLERS[manifest["shortcutHandler"]]
            branch(manifest["id"], handler, manifest, manifest["requiresConfirmation"])
    workflow.speak(workflow.text("This action isn't available in this shortcut.", "Unavailable Action"))


def make_router(team_id):
    workflow = Workflow(team_id)
    workflow.comment("Dictate a request, continue the same Controllr thread, and run the proposed phone action.")
    workflow.comment("Shortcuts generated by Shortcuts Playground. Talk to Controllr by voice.")
    input_ref = Reference({"Type": "ExtensionInput"})
    thread = workflow.text(input_ref.text, "Input Thread ID")
    workflow.set_variable("Thread ID", thread)
    current_thread = Reference({"Type": "Variable", "VariableName": "Thread ID"})

    with workflow.repeat(8):
        utterance = workflow.output("is.workflow.actions.dictatetext", "Dictated Text")
        response = workflow.app_intent("DecideIntent", output="Decision JSON",
                                       transcript=utterance.text, threadID=current_thread.text)
        decision = workflow.output("is.workflow.actions.detect.dictionary", "Decision", WFInput=response.attachment)
        thread_id = workflow.key(decision, "thread_id")
        workflow.set_variable("Thread ID", thread_id)
        response_type = workflow.key(decision, "action")
        summary = workflow.key(decision, "summary")
        with workflow.when(response_type, "clarify") as otherwise:
            workflow.speak(summary)
            otherwise()
            with workflow.when(response_type, "run") as speak_instead:
                dispatch(workflow, decision, summary, thread_id)
                speak_instead()
                workflow.speak(summary)
            workflow.system("exit")

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
