# lowercaseLabs
lowercaseLabs 

## Directory Tree

```text
.
|-- .github
|   `-- workflows
|-- .gitignore
|-- AI_README.md
|-- ClothesPicker
|   |-- .pylintrc
|   |-- CLOTHES_INSTRUCTION.prompt
|   |-- CLOTHES_LIST.prompt
|   |-- README.md
|   |-- Test.ipynb
|   `-- utils.py
|-- README.md
|-- REFACTOR_VIEWS.md
|-- cntx
|   |-- .gitignore
|   |-- README.md
|   |-- backend
|   |-- data
|   |-- frontend
|   `-- package-lock.json
|-- dump
|   `-- Classification.ipynb
|-- gpt_shell
|   `-- gpt.sh
|-- jupyterDocker
|   |-- cpu
|   `-- setup.sh
|-- localOnly
|   |-- .gitignore
|   |-- Makefile
|   |-- README.md
|   |-- backend
|   |-- docker-compose.yml
|   |-- docs
|   |-- ios-app
|   |-- test-apps
|   `-- web-harness
|-- mail
|   |-- .env.example
|   |-- .gitignore
|   |-- IMAP\ access
|   |-- README.md
|   |-- docker
|   |-- docker-compose.yml
|   |-- docs
|   |-- packages
|   |-- pyproject.toml
|   |-- requirements.txt
|   |-- services
|   `-- uv.lock
|-- mapper
|   |-- .gitignore
|   |-- Dockerfile
|   |-- README.md
|   |-- client
|   |-- docker-compose.otp.yml
|   |-- fly.toml
|   |-- otp
|   |-- package-lock.json
|   |-- package.json
|   |-- scripts
|   `-- server
|-- meetUp
|   |-- README.md
|   |-- main.py
|   |-- pyproject.toml
|   `-- uv.lock
|-- metrain
|   |-- README.md
|   `-- imessage_exporter.py
|-- qork
|   |-- .gitignore
|   |-- AGENTS.md
|   |-- README.md
|   |-- build.sh
|   |-- pyproject.toml
|   |-- qork
|   |-- tests
|   |-- upload.sh
|   `-- uv.lock
|-- raggar
|   |-- README.md
|   `-- chatter
|-- research
|   |-- SAP
|   `-- unfiltered
|-- sapProject
|   |-- AAL\ dataset
|   |-- AAVE.jsonl
|   |-- PrelimAnalysis.ipynb
|   `-- WAE.jsonl
|-- theVoidLocal
|   |-- AGENT.md
|   |-- Makefile
|   |-- STATE.md
|   |-- backend
|   |-- docker-compose.yml
|   |-- fly.postgres.toml
|   |-- fly.toml
|   `-- theVoid
|-- trasher
|   |-- .gitignore
|   |-- README.md
|   |-- pyproject.toml
|   |-- src
|   `-- uv.lock
|-- zagency
|   |-- Makefile
|   |-- NEW_FRAMEWORK.md
|   |-- README.md
|   |-- Thesis.md
|   |-- etc
|   |-- pyproject.toml
|   |-- tests
|   `-- zagency
|-- zagent
|   `-- AGENTS.md
|-- zimer
|   |-- README.md
|   |-- pyproject.toml
|   |-- tests
|   |-- zimer
|   `-- zimer.egg-info
|-- znote
|   |-- .notes_config.json
|   |-- README.md
|   |-- docs
|   |-- pyproject.toml
|   |-- src
|   |-- test_bh
|   |-- tests
|   |-- text.md
|   `-- uv.lock
|-- zubillow
|   |-- 500_limit_sf.json
|   |-- README.md
|   |-- main.py
|   |-- manual_zillow.py
|   |-- pyproject.toml
|   |-- rentalcast_api.py
|   |-- uv.lock
|   `-- zillow-mcp-server
`-- zudget
    |-- .gitignore
    |-- AGENT.md
    |-- Dockerfile
    |-- README.md
    |-- backend
    |-- data
    |-- docker-compose.yml
    `-- frontend

65 directories, 87 files
```

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
