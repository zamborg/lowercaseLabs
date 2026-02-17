import cors from "@fastify/cors";
import fastifyStatic from "@fastify/static";
import Fastify from "fastify";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { bboxForPoints, clamp, haversineMeters } from "./geo.mjs";
import { getIsochroneFeatureCollection } from "./isochrone/index.mjs";
import { sourcesToTargets, SUPPORTED_MODES as VALHALLA_MODES } from "./valhalla.mjs";
import { warpGridTwoAnchors } from "./warp.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const clientDist = path.resolve(__dirname, "../../client/dist");

const app = Fastify({ logger: true });
await app.register(cors, { origin: true });

const APP_MODES = new Set(["auto", "bicycle", "pedestrian", "transit"]);

app.get("/api/health", async () => ({ ok: true }));

app.get("/api/geocode", async (req, reply) => {
  const q = `${req.query?.q || ""}`.trim();
  if (!q) return reply.code(400).send({ error: "Missing q" });

  const url = new URL("https://nominatim.openstreetmap.org/search");
  url.searchParams.set("format", "jsonv2");
  url.searchParams.set("limit", "5");
  url.searchParams.set("q", q);

  const res = await fetch(url, {
    headers: {
      "User-Agent": process.env.NOMINATIM_USER_AGENT || "mapper-distorter/0.1 (prototype)",
      "Accept-Language": "en",
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    return reply.code(502).send({ error: `Geocode upstream failed (${res.status})`, detail: text });
  }
  const data = await res.json();
  const results = (Array.isArray(data) ? data : []).map((r) => ({
    displayName: r.display_name,
    lat: Number(r.lat),
    lon: Number(r.lon),
  }));
  return { results };
});

app.get("/api/isochrone", async (req, reply) => {
  const mode = `${req.query?.mode || ""}`;
  const lat = Number(req.query?.lat);
  const lon = Number(req.query?.lon);
  const cutoffs = `${req.query?.cutoffs || ""}`.trim();

  if (!APP_MODES.has(mode)) return reply.code(400).send({ error: "Unsupported mode" });
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return reply.code(400).send({ error: "Invalid coordinates" });
  }
  if (!cutoffs) return reply.code(400).send({ error: "Missing cutoffs" });

  const cutoffsMin = cutoffs
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && n > 0)
    .map((n) => clamp(n, 1, 240));

  if (!cutoffsMin.length) return reply.code(400).send({ error: "Invalid cutoffs" });

  try {
    const featureCollection = await getIsochroneFeatureCollection({
      center: { lat, lon },
      mode,
      cutoffsSec: Array.from(new Set(cutoffsMin))
        .sort((a, b) => a - b)
        .map((m) => m * 60),
    });
    return { center: { lat, lon }, mode, featureCollection };
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Isochrone upstream failed";
    if (msg.includes("OTP_BASE_URL")) return reply.code(400).send({ error: msg });
    return reply
      .code(502)
      .send({ error: msg });
  }
});

app.post("/api/warp-grid", async (req, reply) => {
  const body = req.body || {};
  const a = body.a;
  const b = body.b;
  const mode = body.mode;
  const rows = clamp(Number(body.rows ?? 17), 5, 31);
  const cols = clamp(Number(body.cols ?? 17), 5, 31);

  if (!a || !b) return reply.code(400).send("Missing a/b");
  if (!APP_MODES.has(mode)) return reply.code(400).send("Unsupported mode");

  const aLatLon = { lat: Number(a.lat), lon: Number(a.lon) };
  const bLatLon = { lat: Number(b.lat), lon: Number(b.lon) };

  if (![aLatLon.lat, aLatLon.lon, bLatLon.lat, bLatLon.lon].every(Number.isFinite)) {
    return reply.code(400).send("Invalid coordinates");
  }

  const distABMeters = haversineMeters(aLatLon, bLatLon);

  const center = {
    lat: (aLatLon.lat + bLatLon.lat) / 2,
    lon: (aLatLon.lon + bLatLon.lon) / 2,
  };
  const halfSizeMeters = Math.max(2500, Math.min(16000, distABMeters * 1.2));

  const metersPerDegLat = 111132;
  const metersPerDegLon = metersPerDegLat * Math.cos((center.lat * Math.PI) / 180);
  const dLat = halfSizeMeters / metersPerDegLat;
  const dLon = halfSizeMeters / Math.max(1, metersPerDegLon);

  const bbox = {
    minLat: center.lat - dLat,
    maxLat: center.lat + dLat,
    minLon: center.lon - dLon,
    maxLon: center.lon + dLon,
  };

  const originalPoints = [];
  for (let r = 0; r < rows; r++) {
    const t = rows === 1 ? 0.5 : r / (rows - 1);
    const lat = bbox.minLat + t * (bbox.maxLat - bbox.minLat);
    for (let c = 0; c < cols; c++) {
      const u = cols === 1 ? 0.5 : c / (cols - 1);
      const lon = bbox.minLon + u * (bbox.maxLon - bbox.minLon);
      originalPoints.push({ lat, lon });
    }
  }

  let secondsFromA = [];
  let secondsFromB = [];
  let timeABSec = Number.NaN;

  if (VALHALLA_MODES.has(mode)) {
    const matrix = await sourcesToTargets({
      sources: [aLatLon, bLatLon],
      targets: originalPoints,
      costing: mode,
    });

    const rowsOut = matrix?.sources_to_targets;
    if (!Array.isArray(rowsOut) || rowsOut.length < 2) {
      return reply.code(502).send("Routing matrix missing");
    }

    for (let i = 0; i < originalPoints.length; i++) {
      secondsFromA.push(Number(rowsOut[0]?.[i]?.time));
      secondsFromB.push(Number(rowsOut[1]?.[i]?.time));
    }

    const abMatrix = await sourcesToTargets({
      sources: [aLatLon],
      targets: [bLatLon],
      costing: mode,
    });
    timeABSec = Number(abMatrix?.sources_to_targets?.[0]?.[0]?.time);
  } else if (mode === "transit") {
    try {
      const cutoffsSec = Array.from({ length: 12 }, (_, i) => (i + 1) * 5 * 60); // 5..60 min
      const [aIso, bIso] = await Promise.all([
        getIsochroneFeatureCollection({ center: aLatLon, mode, cutoffsSec }),
        getIsochroneFeatureCollection({ center: bLatLon, mode, cutoffsSec }),
      ]);

      secondsFromA = originalPoints.map((p) => estimateSecondsFromContours(aIso, p));
      secondsFromB = originalPoints.map((p) => estimateSecondsFromContours(bIso, p));
      timeABSec = estimateSecondsFromContours(aIso, bLatLon);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Transit isochrone upstream failed";
      if (msg.includes("OTP_BASE_URL")) return reply.code(400).send(msg);
      return reply.code(502).send(msg);
    }
  } else {
    return reply.code(400).send("Unsupported mode");
  }

  const defaultMps =
    mode === "pedestrian" ? 1.4 : mode === "bicycle" ? 4.2 : mode === "auto" ? 12 : 6.0;
  const metersPerSecond =
    Number.isFinite(timeABSec) && timeABSec > 0 ? distABMeters / timeABSec : defaultMps;

  const warpedPoints = warpGridTwoAnchors({
    aLatLon,
    bLatLon,
    originalPointsLatLon: originalPoints,
    secondsFromA,
    secondsFromB,
    metersPerSecond,
  });

  const overallBbox = bboxForPoints([aLatLon, bLatLon, ...originalPoints, ...warpedPoints]);

  return {
    grid: { rows, cols },
    original: originalPoints,
    warped: warpedPoints,
    a: aLatLon,
    b: bLatLon,
    meta: {
      mode,
      timeABSec,
      distanceABMeters: distABMeters,
      metersPerSecond,
      bbox: overallBbox,
    },
  };
});

function estimateSecondsFromContours(featureCollection, pt) {
  let best = Number.POSITIVE_INFINITY;
  for (const f of featureCollection?.features || []) {
    const cutoffSec = Number(f?.properties?.cutoffSec);
    if (!Number.isFinite(cutoffSec)) continue;
    if (pointInPolygonFeature(pt, f)) best = Math.min(best, cutoffSec);
  }
  return best === Number.POSITIVE_INFINITY ? Number.NaN : best;
}

function pointInPolygonFeature(pt, feature) {
  const geom = feature?.geometry;
  if (!geom) return false;
  if (geom.type === "Polygon") return pointInPolygon(pt, geom.coordinates);
  if (geom.type === "MultiPolygon") {
    for (const poly of geom.coordinates || []) {
      if (pointInPolygon(pt, poly)) return true;
    }
  }
  return false;
}

function pointInPolygon(pt, polygonCoordinates) {
  if (!Array.isArray(polygonCoordinates) || polygonCoordinates.length === 0) return false;
  const [outerRing, ...holes] = polygonCoordinates;
  if (!pointInRing(pt, outerRing)) return false;
  for (const hole of holes) {
    if (pointInRing(pt, hole)) return false;
  }
  return true;
}

function pointInRing(pt, ring) {
  if (!Array.isArray(ring) || ring.length < 3) return false;
  const x = pt.lon;
  const y = pt.lat;
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = Number(ring[i]?.[0]);
    const yi = Number(ring[i]?.[1]);
    const xj = Number(ring[j]?.[0]);
    const yj = Number(ring[j]?.[1]);

    const intersects = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 0.0) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

if (fs.existsSync(clientDist)) {
  await app.register(fastifyStatic, {
    root: clientDist,
  });
}

app.setNotFoundHandler(async (req, reply) => {
  if (req.raw.url?.startsWith("/api/")) return reply.code(404).send({ error: "Not found" });
  if (fs.existsSync(path.join(clientDist, "index.html"))) return reply.sendFile("index.html");
  return reply.code(404).send({ error: "Client not built. Run `npm -w client run build`." });
});

const port = Number(process.env.PORT || 8080);
const host = process.env.HOST || "0.0.0.0";

await app.listen({ port, host });
