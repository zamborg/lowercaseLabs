# Sovereign Web Harness

Browser-only harness for testing:
- authentication
- app chart parsing + registration
- route resolution
- Data Plane routing
- web session cookie flow

## Run

```bash
cd web-harness
python3 -m http.server 4173
```

Open [http://localhost:4173](http://localhost:4173).

## Notes

- Use `Set Web Session Cookie` after auth to enable cookie-based browser routing.
- For programmatic calls you can still send `Authorization: Bearer <token>`.
