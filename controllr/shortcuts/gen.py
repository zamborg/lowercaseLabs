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

def router(network=False):
    dictate, decide, dec, name, payload = (U() for _ in range(5))
    intent = "DecideNetworkIntent" if network else "DecideIntent"
    return shortcut([
        act("is.workflow.actions.dictatetext", {"UUID": dictate, "CustomOutputName": "Dictated Text"}),
        act(f"{BUNDLE}.{intent}", {"UUID": decide, "CustomOutputName": "Decision JSON", "transcript": text_ref(dictate, "Dictated Text"),
            "AppIntentDescriptor": {"AppIntentIdentifier": intent, "BundleIdentifier": BUNDLE, "Name": "controllr", "TeamIdentifier": "9G57J2JMF7"}}),
        act("is.workflow.actions.detect.dictionary", {"WFInput": out_ref(decide, "Decision JSON"), "UUID": dec, "CustomOutputName": "Decision"}),
        get_key("shortcut", dec, "Decision", name, "Shortcut Name"),
        get_key("input", dec, "Decision", payload, "Payload"),
        act("is.workflow.actions.runworkflow", {"WFWorkflow": out_ref(name, "Shortcut Name"), "WFInput": out_ref(payload, "Payload"), "WFShowWorkflow": False}),
    ])

SHORTCUTS = {"Controllr · Spike": spike_module(), "Controllr · Speak": speak_module(), "Controllr · Router": router(), "Controllr · Router (network)": router(True)}

for n, s in SHORTCUTS.items():
    src = HERE / "src" / f"{n}.shortcut.xml"
    src.write_bytes(plistlib.dumps(s, fmt=plistlib.FMT_XML))
    raw = HERE / "signed" / f"unsigned-{n}.shortcut"
    raw.write_bytes(plistlib.dumps(s, fmt=plistlib.FMT_BINARY))
    r = subprocess.run(["shortcuts", "sign", "--mode", "anyone", "--input", str(raw), "--output", str(HERE / "signed" / f"{n}.shortcut")], capture_output=True, text=True)
    raw.unlink()
    print(n, "OK" if r.returncode == 0 else f"FAIL: {r.stderr.strip()[:300]}")
