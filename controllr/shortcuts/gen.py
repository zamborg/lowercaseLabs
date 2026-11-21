#!/usr/bin/env python3
"""Generate Controllr shortcut XML plists (src/) and sign them (signed/). macOS only."""
import plistlib, subprocess, uuid, sys, pathlib

HERE = pathlib.Path(__file__).parent
BUNDLE = "com.lowercaselabs.controllr"
U = lambda: str(uuid.uuid4()).upper()

def out_ref(uid, name):
    return {"Value": {"OutputUUID": uid, "OutputName": name, "Type": "ActionOutput"},
            "WFSerializationType": "WFTextTokenAttachment"}

def text_ref(uid, name):
    return {"Value": {"string": "￼",
                      "attachmentsByRange": {"{0, 1}": {"OutputUUID": uid, "OutputName": name, "Type": "ActionOutput"}}},
            "WFSerializationType": "WFTextTokenString"}

def act(ident, params=None):
    return {"WFWorkflowActionIdentifier": ident, "WFWorkflowActionParameters": params or {}}

def get_key(key, src_uid, src_name, uid, name):
    return act("is.workflow.actions.getvalueforkey",
               {"WFDictionaryKey": key, "WFInput": {"Value": {"OutputUUID": src_uid, "OutputName": src_name, "Type": "ActionOutput"},
                "WFSerializationType": "WFTextTokenAttachment"}, "UUID": uid, "CustomOutputName": name})

def shortcut(actions):
    return {"WFWorkflowActions": actions, "WFWorkflowClientVersion": "2900", "WFWorkflowMinimumClientVersion": 900,
            "WFWorkflowIcon": {"WFWorkflowIconStartColor": 463140863, "WFWorkflowIconGlyphNumber": 59511},
            "WFWorkflowImportQuestions": [], "WFWorkflowInputContentItemClasses": [], "WFWorkflowTypes": []}

def spike_module():
    d = U(); v = U()
    return shortcut([
        act("is.workflow.actions.detect.dictionary", {"WFInput": {"Value": {"Type": "ExtensionInput"}, "WFSerializationType": "WFTextTokenAttachment"}, "UUID": d, "CustomOutputName": "Dict"}),
        get_key("echo", d, "Dict", v, "Echo"),
        act("is.workflow.actions.alert", {"WFAlertActionTitle": "Spike", "WFAlertActionMessage": text_ref(v, "Echo"), "WFAlertActionCancelButtonShown": False}),
    ])

def speak_module():
    d = U(); v = U()
    return shortcut([
        act("is.workflow.actions.detect.dictionary", {"WFInput": {"Value": {"Type": "ExtensionInput"}, "WFSerializationType": "WFTextTokenAttachment"}, "UUID": d, "CustomOutputName": "Dict"}),
        get_key("speak", d, "Dict", v, "Speak"),
        act("is.workflow.actions.speaktext", {"WFText": text_ref(v, "Speak")}),
    ])

def router(network=False, text=False):
    """No Run Shortcut (it prompts every run). Modules are inline If branches on the module id."""
    dictate, decide, dec, name, payload, echo, speak, g, modtext = (U() for _ in range(9))
    intent = "DecideNetworkIntent" if network else "DecideIntent"
    ref = {"Type": "Variable", "Variable": out_ref(modtext, "Module Text")}
    return shortcut([
        (act("is.workflow.actions.gettext", {"WFTextActionText": "hello", "UUID": dictate, "CustomOutputName": "Dictated Text"}) if text
         else act("is.workflow.actions.dictatetext", {"UUID": dictate, "CustomOutputName": "Dictated Text"})),
        act(f"{BUNDLE}.{intent}", {"UUID": decide, "CustomOutputName": "Decision JSON", "transcript": text_ref(dictate, "Dictated Text"),
            "AppIntentDescriptor": {"AppIntentIdentifier": intent, "BundleIdentifier": BUNDLE, "Name": "controllr", "TeamIdentifier": "9G57J2JMF7"}}),
        act("is.workflow.actions.detect.dictionary", {"WFInput": out_ref(decide, "Decision JSON"), "UUID": dec, "CustomOutputName": "Decision"}),
        get_key("shortcut", dec, "Decision", name, "Module"),
        get_key("input", dec, "Decision", payload, "Payload"),
        act("is.workflow.actions.gettext", {"WFTextActionText": text_ref(name, "Module"), "UUID": modtext, "CustomOutputName": "Module Text"}),
        act("is.workflow.actions.conditional", {"WFControlFlowMode": 0, "GroupingIdentifier": g, "WFInput": ref, "WFCondition": 4, "WFConditionalActionString": "Controllr · Spike"}),
        get_key("echo", payload, "Payload", echo, "Echo"),
        act("is.workflow.actions.alert", {"WFAlertActionTitle": "Spike", "WFAlertActionMessage": text_ref(echo, "Echo"), "WFAlertActionCancelButtonShown": False}),
        act("is.workflow.actions.conditional", {"GroupingIdentifier": g, "WFControlFlowMode": 1}),
        get_key("speak", payload, "Payload", speak, "Speak"),
        (act("is.workflow.actions.alert", {"WFAlertActionTitle": "Result", "WFAlertActionMessage": text_ref(speak, "Speak"), "WFAlertActionCancelButtonShown": False}) if text
         else act("is.workflow.actions.speaktext", {"WFText": text_ref(speak, "Speak")})),
        act("is.workflow.actions.conditional", {"GroupingIdentifier": g, "WFControlFlowMode": 2}),
    ])

def router_static_test():
    dictate, decide, dec, payload = (U() for _ in range(4))
    return shortcut([
        act("is.workflow.actions.dictatetext", {"UUID": dictate, "CustomOutputName": "Dictated Text"}),
        act(f"{BUNDLE}.DecideIntent", {"UUID": decide, "CustomOutputName": "Decision JSON", "transcript": text_ref(dictate, "Dictated Text"),
            "AppIntentDescriptor": {"AppIntentIdentifier": "DecideIntent", "BundleIdentifier": BUNDLE, "Name": "controllr", "TeamIdentifier": "9G57J2JMF7"}}),
        act("is.workflow.actions.detect.dictionary", {"WFInput": out_ref(decide, "Decision JSON"), "UUID": dec, "CustomOutputName": "Decision"}),
        get_key("input", dec, "Decision", payload, "Payload"),
        act("is.workflow.actions.runworkflow", {"WFWorkflow": {"workflowName": "Controllr · Spike", "isSelf": False}, "WFInput": out_ref(payload, "Payload"), "WFShowWorkflow": False}),
    ])


def if_test(variant):
    txt, g = U(), U()
    ref = {"Type": "Variable", "Variable": out_ref(txt, "Text")}
    if variant == "A":
        cond = {"WFControlFlowMode": 0, "GroupingIdentifier": g, "WFInput": ref, "WFConditionalActionString": "run",
                "WFConditions": {"WFSerializationType": "WFContentPredicateTableTemplate",
                 "Value": {"WFActionParameterFilterPrefix": 1, "WFActionParameterFilterTemplates": [
                  {"WFCondition": 4, "Operator": 4, "Removable": True, "WFInput": ref,
                   "WFConditionalActionString": "run", "Values": {"String": "run"}}]}}}
    else:
        cond = {"WFControlFlowMode": 0, "GroupingIdentifier": g, "WFInput": ref, "WFCondition": 4, "WFConditionalActionString": "run"}
    alert = lambda m: act("is.workflow.actions.alert", {"WFAlertActionTitle": "If test", "WFAlertActionMessage": m, "WFAlertActionCancelButtonShown": False})
    return shortcut([
        act("is.workflow.actions.gettext", {"WFTextActionText": "run", "UUID": txt, "CustomOutputName": "Text"}),
        act("is.workflow.actions.conditional", cond), alert("yes"),
        act("is.workflow.actions.conditional", {"GroupingIdentifier": g, "WFControlFlowMode": 1}), alert("no"),
        act("is.workflow.actions.conditional", {"GroupingIdentifier": g, "WFControlFlowMode": 2}),
    ])

SHORTCUTS = {"Controllr · Router": router(), "Controllr · Router (network)": router(True), "Controllr · Alt": router(False, True), "Controllr · Alt (network)": router(True, True)}

for n, s in SHORTCUTS.items():
    src = HERE / "src" / f"{n}.shortcut.xml"
    src.write_bytes(plistlib.dumps(s, fmt=plistlib.FMT_XML))
    raw = HERE / "signed" / f"unsigned-{n}.shortcut"
    raw.write_bytes(plistlib.dumps(s, fmt=plistlib.FMT_BINARY))
    r = subprocess.run(["shortcuts", "sign", "--mode", "anyone", "--input", str(raw), "--output", str(HERE / "signed" / f"{n}.shortcut")], capture_output=True, text=True)
    raw.unlink()
    print(n, "OK" if r.returncode == 0 else f"FAIL: {r.stderr.strip()[:300]}")
