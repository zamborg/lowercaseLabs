# lowercaseLabs
lowercaseLabs 

## On Update Frequency:

I push code much more frequently than I push readme's. I attempt to maintain decent README into this repo by using AI_README.md --> an AI generated readme. 

## gpt.sh
use `$gpt XXX` to use OpenAI's gpt3.5 for chat completion with a question. Nice to have in the commandline for debugging!

To use install jq:
`brew install jq`

### Tips?
- I highly recommend updating the prompt to make it as useful as possible for you.
  - I have my system information, sometimes I'll add what language I'm working in.
  - If you're having fun you can also play with branching statements in bash to give yourself different default prompts before your query gets passed in.
- I also recommend adding `-s` to the curl command if you don't like the curl load bar.
- Remove the linebreak if you have better eyes than I do 👀

## Directory Structure

```
.
├── AI_README.md
├── ClothesPicker
│   ├── CLOTHES_INSTRUCTION.prompt
│   ├── CLOTHES_LIST.prompt
│   ├── README.md
│   ├── Test.ipynb
│   └── utils.py
├── README.md
├── REFACTOR_VIEWS.md
├── cntx
│   ├── README.md
│   ├── backend
│   │   ├── ai.py
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── data
│   └── frontend
│       ├── README.md
│       ├── eslint.config.js
│       ├── index.html
│       ├── package.json
│       ├── public
│       └── src
│           ├── App.css
│           ├── App.tsx
│           ├── api.ts
│           ├── assets
│           ├── components
│           │   ├── GraphView.tsx
│           │   ├── NoteCard.tsx
│           │   ├── NoteComposer.tsx
│           │   ├── NoteList.tsx
│           │   ├── SearchPanel.tsx
│           │   ├── TagInput.tsx
│           │   └── TagPanel.tsx
│           ├── index.css
│           ├── main.tsx
│           └── types.ts
├── dump
│   └── Classification.ipynb
├── gpt_shell
│   └── gpt.sh
├── jupyterDocker
│   ├── cpu
│   │   ├── Dockerfile
│   │   ├── docker_push.sh
│   │   ├── jupyter_notebook_config.py
│   │   ├── requirements.txt
│   │   └── run.sh
│   └── setup.sh
├── localOnly
│   ├── Makefile
│   ├── README.md
│   ├── backend
│   ├── docker-compose.yml
│   ├── docs
│   ├── ios-app
│   ├── test-apps
│   └── web-harness
├── mail
│   ├── README.md
│   ├── docker
│   ├── docker-compose.yml
│   ├── docs
│   ├── packages
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── services
│   │   ├── agent
│   │   ├── api
│   │   ├── mail-sync
│   │   ├── policy
│   │   └── tui
│   └── uv.lock
├── mapper
│   ├── Dockerfile
│   ├── README.md
│   ├── client
│   ├── docker-compose.otp.yml
│   ├── fly.toml
│   ├── otp
│   ├── scripts
│   └── server
├── meetUp
│   ├── README.md
│   ├── main.py
│   ├── pyproject.toml
│   └── uv.lock
├── metrain
│   ├── README.md
│   └── imessage_exporter.py
├── qork
│   ├── AGENTS.md
│   ├── README.md
│   ├── build.sh
│   ├── pyproject.toml
│   ├── qork
│   ├── tests
│   ├── upload.sh
│   └── uv.lock
├── raggar
│   ├── README.md
│   └── chatter
├── research
│   ├── SAP
│   └── unfiltered
├── sapProject
│   ├── AAL dataset
│   ├── AAVE.jsonl
│   ├── PrelimAnalysis.ipynb
│   └── WAE.jsonl
├── theVoidLocal
│   ├── AGENT.md
│   ├── Makefile
│   ├── STATE.md
│   ├── backend
│   ├── docker-compose.yml
│   ├── fly.postgres.toml
│   ├── fly.toml
│   └── theVoid
├── trasher
│   ├── README.md
│   ├── pyproject.toml
│   ├── src
│   └── uv.lock
├── zagency
│   ├── Makefile
│   ├── README.md
│   ├── Thesis.md
│   ├── etc
│   ├── pyproject.toml
│   ├── tests
│   └── zagency
├── zagent
│   └── AGENTS.md
├── zimer
│   ├── README.md
│   ├── dist
│   ├── pyproject.toml
│   ├── tests
│   └── zimer
├── znote
│   ├── README.md
│   ├── docs
│   ├── pyproject.toml
│   ├── src
│   ├── tests
│   └── uv.lock
├── zubillow
│   ├── README.md
│   ├── main.py
│   ├── pyproject.toml
│   └── zillow-mcp-server
└── zudget
    ├── AGENT.md
    ├── Dockerfile
    ├── README.md
    ├── backend
    ├── data
    ├── docker-compose.yml
    └── frontend
```
