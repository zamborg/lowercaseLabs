#!/usr/bin/env python3
"""Build the modular Controllr Voice router and its independently runnable handlers."""
import argparse
import json
from pathlib import Path
import plistlib

from gen import HANDLERS, HERE, MANIFESTS, Reference, Workflow, authorize, make_router, open_url

VOICE_NAME = "Controllr Voice 2.1"
ACTION_PREFIX = "Controllr Action "
HANDOFFS = ("open_url", "open_url_confirm")


def dispatch(workflow, decision, summary, thread_id):
    action = workflow.key(decision, "shortcut")
    action_id = workflow.key(decision, "action_id")
    payload = workflow.key(decision, "input", as_text=False)
    confirmation = workflow.key(decision, "confirmation")
    callback = {"threadID": thread_id.text, "actionID": action_id.text}
    authorization = authorize(workflow, confirmation, summary, callback)

    with workflow.when(authorization, "ready") as denied:
        with workflow.when(action, "approved_native") as run_shortcut:
            # The app executes authorized native writes in the completion callback.
            workflow.set_variable("Action Result", workflow.text("ok", "Native Action"))
            run_shortcut()
            handler = workflow.text({
                "WFSerializationType": "WFTextTokenString",
                "Value": {
                    "string": ACTION_PREFIX + "\ufffc",
                    "attachmentsByRange": {f"{{{len(ACTION_PREFIX)}, 1}}": action.value},
                },
            }, "Registered Handler")
            result = workflow.output("is.workflow.actions.runworkflow", "Handler Result",
                                     WFWorkflow=handler.attachment, WFInput=payload.attachment, WFShowWorkflow=False)
            workflow.set_variable("Action Result", result)
        response = workflow.app_intent("CompleteActionIntent", result_name="Next Decision", **callback,
                                       output=Reference.variable("Action Result").text)
        workflow.set_variable("Decision JSON", response)
        workflow.set_variable("Next Step", workflow.text("dispatch", "Handle Result"))
        denied()
        workflow.speak(authorization)
        workflow.system("exit")


def make_handler(team_id, identifier, build, manifest=None):
    workflow = Workflow(team_id)
    shortcut_input = Reference({"Type": "ExtensionInput"})
    payload = workflow.output("is.workflow.actions.detect.dictionary", "Handler Input", WFInput=shortcut_input.attachment)
    result = build(workflow, payload, manifest)
    if result is None:
        result = workflow.text("ok", "Action Completed")
    workflow.system("output", WFOutput=result.text, WFResponse=result.text)
    return workflow.document(ACTION_PREFIX + identifier, ("WFDictionaryContentItem", "WFStringContentItem"))


def make_workflows(team_id):
    for path in sorted(MANIFESTS.glob("*.json")):
        manifest = json.loads(path.read_text())
        if manifest["kind"] == "shortcut":
            yield make_handler(team_id, manifest["id"], HANDLERS[manifest["shortcutHandler"]], manifest)
    for identifier in HANDOFFS:
        yield make_handler(team_id, identifier, open_url)
    yield make_router(team_id, name=VOICE_NAME, dispatch_action=dispatch, router_version=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "build" / "modular-shortcuts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for document in make_workflows(args.team_id):
        name = document["WFWorkflowName"]
        path = args.output_dir / f"{name}.xml"
        path.write_bytes(plistlib.dumps(document))
        print(f"{len(document['WFWorkflowActions']):3} steps  {path}")


if __name__ == "__main__":
    main()
