# Sovereign Agent Playbook

This folder is the implementation playbook for coding agents that need to create sovereign apps compatible with this platform.

## Read Order (Required)

1. [01-system-contract.md](./01-system-contract.md)
2. [02-app-chart-spec.md](./02-app-chart-spec.md)
3. Choose one:
   - [03-web-app-playbook.md](./03-web-app-playbook.md)
   - [04-api-backend-playbook.md](./04-api-backend-playbook.md)
4. [05-delivery-checklist.md](./05-delivery-checklist.md)
5. [06-demo-runbook.md](./06-demo-runbook.md)

## Goal For Agents

Produce repositories that can be ingested by:
- `POST /apps/charts/parse`
- `POST /apps/from-chart`

And provisioned by:
- `POST /pods/provision`

Without requiring platform-specific changes.

## Platform Expectations Summary

- Every app repo must include `Dockerfile` and `fly.toml`.
- Every app must listen on a single HTTP port (default: 8080).
- `fly.toml` must declare `networking_mode` in metadata (`proxy` or `direct`).
- App code must assume one pod = one isolated tenant/friend-group backend.

## Templates and Examples

- API-focused example: `test-apps/echo-pod`
- Web-focused example: `test-apps/web-pod`
- Prompt bootstrap: [agent-prompt-template.md](./agent-prompt-template.md)
