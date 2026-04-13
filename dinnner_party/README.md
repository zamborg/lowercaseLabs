# Dinner Party

This is a tiny local chat room for a Sunday dinner-planning simulation.

The model is now:

- `people/<id>/person.json`: structured runtime metadata created from the UI.
- `people/<id>/soul.md`: the canonical prompt-like identity for that person.
- `config/calendar-connector.json`: how calendar is meant to be sourced. Right now it is modeled as an MCP-backed Google Calendar connector with a mock fallback snapshot.
- `data/calendar.json`: the current fallback calendar snapshot.

The UI is intentionally tight. It lets each person register an agent, stores those agents as files, and then runs the current set with a model string you choose.

## Run it

```bash
python3 server.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Demo flow

1. Open the app.
2. Each person fills out the agent form with display name, id, Google Calendar MCP server name, calendar identity, and their soul prompt.
3. Press `Save Agent`.
4. Press `Connect MCP` on that agent card. The UI starts `codex mcp login`, opens the OAuth URL, and waits for the browser callback on this machine.
5. Repeat for the other agents, using different browser profiles if you want different Google accounts.
6. Type the model you want in the run bar.
7. Press `Run Registered Agents`.

That button hits `POST /api/agent_runs/start` and launches one `codex exec` run per currently registered person.

The UI also lets you:

- reset the room,
- stop the current run,
- inspect each person's soul,
- trigger per-agent MCP auth from the browser,
- inspect the MCP connector status for each registered agent.

## Run all four participants

```bash
./test.sh
```

This script:

1. Starts `server.py` if it is not already running.
2. Resets the chat room.
3. Builds four participant-specific prompts from [task.md](/Users/zubinaysola/Documents/personal/lowercaseLabs/dinnner_party/task.md) plus each person's `soul.md`.
4. Launches one background `codex exec` run for each of `zubin`, `griffin`, `nico`, and `patrick` using `gpt-5.1` by default.
5. Opens the UI in your browser on macOS when possible.

Each participant writes its terminal output to `logs/<name>.log`.

Override the model if you want:

```bash
MODEL=gpt-5.1 ./test.sh
```

## HTTP primitives

### Read unread messages

```bash
curl "http://127.0.0.1:8000/api/read_messages?after=0"
```

### Send a message

```bash
curl -X POST "http://127.0.0.1:8000/api/send_message" \
  -H "Content-Type: application/json" \
  -d '{"sender":"zubin","text":"I can do 7:30 if we pick a place quickly."}'
```

### Reset the room

```bash
curl -X POST "http://127.0.0.1:8000/api/reset"
```

### Fetch the shared calendar

```bash
curl "http://127.0.0.1:8000/api/calendar"
```

### Fetch one soul through the legacy endpoint

```bash
curl "http://127.0.0.1:8000/api/context_packs/zubin"
```

Legacy only. The preferred endpoint is:

```bash
curl "http://127.0.0.1:8000/api/people/zubin"
```

### Fetch the calendar connector

```bash
curl "http://127.0.0.1:8000/api/calendar_connector"
```

### Fetch the API spec

```bash
curl "http://127.0.0.1:8000/api/spec"
```

## Person model

The current person data model is documented in [docs/person-model.md](/Users/zubinaysola/Documents/personal/lowercaseLabs/dinnner_party/docs/person-model.md).

## `codex exec` shape

The shared instructions live in [task.md](/Users/zubinaysola/Documents/personal/lowercaseLabs/dinnner_party/task.md). `test.sh` substitutes the participant name and model into that template, then appends the matching `people/<id>/soul.md` content before feeding it into `codex exec`.

If you want to run just one participant manually, use the same shape:

```bash
MODEL=gpt-5.1
AGENT=griffin
SOUL_PATH=/Users/zubinaysola/Documents/personal/lowercaseLabs/dinnner_party/people/$AGENT/soul.md
PERSON_PATH=/Users/zubinaysola/Documents/personal/lowercaseLabs/dinnner_party/people/$AGENT/person.json

{
  sed \
    -e "s/__AGENT__/$AGENT/g" \
    -e "s/__MODEL__/$MODEL/g" \
    -e "s#__SOUL_PATH__#$SOUL_PATH#g" \
    task.md
  echo
  echo "# Person Record"
  echo
  echo '```json'
  cat "$PERSON_PATH"
  echo
  echo '```'
  echo
  echo "# Soul"
  echo
  cat "$SOUL_PATH"
} | \
  codex exec \
    --skip-git-repo-check \
    --dangerously-bypass-approvals-and-sandbox \
    -m "$MODEL" \
    -C /Users/zubinaysola/Documents/personal/lowercaseLabs/dinnner_party
```
