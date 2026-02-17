# Mapper (Transit Distortion)

Prototype webapp to visualize “time-distance”:

- Warp grid: warps map geometry based on travel-time distance between two anchors (Home/Work).
- Isochrone overlay: draws travel-time isochrones and lets you drag/superimpose them (inspired by thetruesize.com).

## Dev

```bash
npm install
npm run dev
```

Open the Vite URL printed in your terminal (usually `http://localhost:5173`). If you see “Port 5173 is in use, trying another one…”, use the new port it prints.

If things look blank or “flash”, you probably have a stale process holding ports; reset them with:

```bash
npm run dev:reset-ports
```

## Build / run

```bash
npm run build
npm run start
```

Server listens on `http://localhost:8080`.

## Transit (SF starter, local OpenTripPlanner)

This wires up **real transit** travel times using a local OpenTripPlanner (OTP) container.

```bash
npm run otp:download
npm run otp:build
npm run dev:otp
```

Then open `http://localhost:5173` and set `Mode = transit`.

Notes:

- OTP runs on `http://localhost:8081` (the app server calls it via `OTP_BASE_URL=http://localhost:8081`).
- The default SF starter downloads: SF OSM extract + BART GTFS. Add more GTFS zips into `otp/data/` to expand coverage.

## Notes

- Routing backend (auto/bike/walk): Valhalla demo (`valhalla1.openstreetmap.de`) via `POST /api/warp-grid` and `GET /api/isochrone`.
- Transit: no signup required — you run your own OpenTripPlanner (OTP) instance. Set `OTP_BASE_URL` (and optionally `OTP_ROUTER_ID`).
- Geocoding: Nominatim (`/api/geocode`).
- For `transit` warp-grid, travel times are approximated from isochrone contours (point-in-polygon), so results are intentionally “map-art” rather than precise routing analytics.

## Deploy (Fly.io)

```bash
fly launch
fly deploy
```

If you keep the provided `fly.toml`, set `app = "CHANGE_ME"` to your Fly app name first.
