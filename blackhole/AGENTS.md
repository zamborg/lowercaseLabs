# blackhole — dev guide

## Test

```bash
cd blackhole/backend
python -m pytest tests/ -v
```

Live OpenAI smoke test (requires real API key):

```bash
BLACKHOLE_RUN_OPENAI_TESTS=1 OPENAI_API_KEY=<key> python -m pytest tests/ -v -k smoke
```

## Commit & push

```bash
cd <repo root>          # lowercaseLabs/
git add blackhole/
git commit -m "your message"
git push
```

## Deploy

```bash
cd blackhole/backend
fly deploy
```

First-time setup only:

```bash
fly volumes create blackhole_data --size 1
fly secrets set JWT_SECRET=<secret> OPENAI_API_KEY=<key> ADMIN_USERNAME=<user> ADMIN_PASSWORD=<pass>
```

App name: `blackhole` → deploys to `https://blackhole.fly.dev`
