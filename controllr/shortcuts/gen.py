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


def tok(uid, name):
    return {"Value": {"OutputUUID": uid, "OutputName": name, "Type": "ActionOutput"}, "WFSerializationType": "WFTextTokenAttachment"}

def iff(val_uid, val_name, equals, body):
    g = U()
    ref = {"Type": "Variable", "Variable": out_ref(val_uid, val_name)}
    return ([act("is.workflow.actions.conditional", {"WFControlFlowMode": 0, "GroupingIdentifier": g, "WFInput": ref, "WFCondition": 4, "WFConditionalActionString": equals})]
            + body + [act("is.workflow.actions.conditional", {"GroupingIdentifier": g, "WFControlFlowMode": 2})])

def router(text=None):
    """Voice router (text=None) or text-input test router (text=phrase). No Run Shortcut: inline If branches."""
    A = []
    def key_text(key, src, srcname):
        k, t = U(), U()
        A.append(get_key(key, src, srcname, k, key))
        A.append(act("is.workflow.actions.gettext", {"WFTextActionText": text_ref(k, key), "UUID": t, "CustomOutputName": key.title()}))
        return t, key.title()
    dictate, decide, dec, payload, mod = (U() for _ in range(5))
    A.append(act("is.workflow.actions.gettext", {"WFTextActionText": text, "UUID": dictate, "CustomOutputName": "Dictated Text"}) if text
             else act("is.workflow.actions.dictatetext", {"UUID": dictate, "CustomOutputName": "Dictated Text"}))
    A.append(act(f"{BUNDLE}.DecideIntent", {"UUID": decide, "CustomOutputName": "Decision JSON", "transcript": text_ref(dictate, "Dictated Text"),
        "AppIntentDescriptor": {"AppIntentIdentifier": "DecideIntent", "BundleIdentifier": BUNDLE, "Name": "controllr", "TeamIdentifier": "9G57J2JMF7"}}))
    A.append(act("is.workflow.actions.detect.dictionary", {"WFInput": out_ref(decide, "Decision JSON"), "UUID": dec, "CustomOutputName": "Decision"}))
    A.append(get_key("input", dec, "Decision", payload, "Payload"))
    modt, modn = key_text("shortcut", dec, "Decision")
    summ, summn = key_text("summary", dec, "Decision")
    def pk(key):
        k, t = U(), U()
        A.append(get_key(key, payload, "Payload", k, key))
        A.append(act("is.workflow.actions.gettext", {"WFTextActionText": text_ref(k, key), "UUID": t, "CustomOutputName": key.title()}))
        return t, key.title()
    say = lambda t, n: (act("is.workflow.actions.alert", {"WFAlertActionTitle": "controllr", "WFAlertActionMessage": text_ref(t, n), "WFAlertActionCancelButtonShown": False}) if text
                        else act("is.workflow.actions.speaktext", {"WFText": text_ref(t, n)}))
    confirm = lambda: act("is.workflow.actions.alert", {"WFAlertActionTitle": "controllr", "WFAlertActionMessage": text_ref(summ, summn), "WFAlertActionCancelButtonShown": True})
    # speak
    st, sn = pk("speak"); A += iff(modt, modn, "speak", [say(st, sn)])
    # open_url / open_url_confirm
    ut, un = pk("url")
    A += iff(modt, modn, "open_url", [act("is.workflow.actions.openurl", {"WFInput": text_ref(ut, un)})])
    A += iff(modt, modn, "open_url_confirm", [confirm(), act("is.workflow.actions.openurl", {"WFInput": text_ref(ut, un)})])
    # send_message
    bt, bn = pk("body"); rt, rn = pk("recipient")
    A += iff(modt, modn, "send_message", [confirm(), act("is.workflow.actions.sendmessage",
        {"WFSendMessageContent": text_ref(bt, bn), "WFSendMessageActionRecipients": text_ref(rt, rn), "ShowWhenRun": False})])
    # media_control
    ct, cn = pk("command")
    media = (iff(ct, cn, "play", [act("is.workflow.actions.pausemusic", {"WFPlayPauseBehavior": "Play"})])
           + iff(ct, cn, "pause", [act("is.workflow.actions.pausemusic", {"WFPlayPauseBehavior": "Pause"})])
           + iff(ct, cn, "next", [act("is.workflow.actions.skipforward")])
           + iff(ct, cn, "previous", [act("is.workflow.actions.skipback")]))
    A += iff(modt, modn, "media_control", media)
    # start_timer: Decide supplies total seconds in "seconds_total"
    tt, tn = pk("seconds_total")
    # Quantity fields can't take a variable (shows "Ask each time"), so enumerate durations.
    secs = [10, 15, 20, 30, 45] + [60 * m for m in range(1, 121)]
    def timer(n):
        u, m = ("min", n // 60) if n % 60 == 0 else ("sec", n)
        return act("is.workflow.actions.timer.start", {"WFDuration": {"Value": {"Unit": u, "Magnitude": m}, "WFSerializationType": "WFQuantityFieldValue"}})
    body = [x for n in secs for x in iff(tt, tn, str(n), [timer(n)])]
    A += iff(modt, modn, "start_timer", body)
    # set_focus
    ft, fn = pk("focus")
    focus = (iff(ft, fn, "dnd", [act("is.workflow.actions.dnd.set", {"Enabled": 1})])
           + iff(ft, fn, "sleep", [act("is.workflow.actions.dnd.set", {"Enabled": 1, "FocusModes": {"DisplayString": "Sleep", "Identifier": "com.apple.sleep.sleep-mode"}})])
           + iff(ft, fn, "work", [act("is.workflow.actions.dnd.set", {"Enabled": 1, "FocusModes": {"DisplayString": "Work", "Identifier": "com.apple.focus.work"}})])
           + iff(ft, fn, "off", [act("is.workflow.actions.dnd.set", {"Enabled": 0})]))
    A += iff(modt, modn, "set_focus", focus)
    # flashlight
    lt, ln = pk("on")
    A += iff(modt, modn, "flashlight", iff(lt, ln, "0", [act("is.workflow.actions.flashlight", {"WFFlashlightSetting": "Off"})])
                                      + iff(lt, ln, "1", [act("is.workflow.actions.flashlight", {"WFFlashlightSetting": "On"})]))
    # not yet supported via shortcut actions (see DECISIONS.md)
    for m in ("set_alarm", "create_note"):
        n = U()
        A += iff(modt, modn, m, [act("is.workflow.actions.gettext", {"WFTextActionText": "That isn't set up yet.", "UUID": n, "CustomOutputName": "Msg"}), say(n, "Msg")])
    return shortcut(A)

import sys
PHRASES = {"Txt": "what time is it in Tokyo", "Timer": "set a timer for 5 minutes", "Focus": "turn on do not disturb", "Text": "text mom I'm running late"}
SHORTCUTS = {"Controllr · Router": router()}
SHORTCUTS.update({f"Controllr · Test {k}": router(v) for k, v in PHRASES.items()})

for n, s in SHORTCUTS.items():
    src = HERE / "src" / f"{n}.shortcut.xml"
    src.write_bytes(plistlib.dumps(s, fmt=plistlib.FMT_XML))
    raw = HERE / "signed" / f"unsigned-{n}.shortcut"
    raw.write_bytes(plistlib.dumps(s, fmt=plistlib.FMT_BINARY))
    r = subprocess.run(["shortcuts", "sign", "--mode", "anyone", "--input", str(raw), "--output", str(HERE / "signed" / f"{n}.shortcut")], capture_output=True, text=True)
    raw.unlink()
    print(n, "OK" if r.returncode == 0 else f"FAIL: {r.stderr.strip()[:300]}")
