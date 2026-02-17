const VALHALLA_BASE = process.env.VALHALLA_BASE_URL || "https://valhalla1.openstreetmap.de";

export async function valhallaIsochroneFeatureCollection({ center, costing, cutoffsSec }) {
  const contours = cutoffsSec.map((s) => ({ time: s / 60 }));
  const res = await fetch(`${VALHALLA_BASE}/isochrone`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      locations: [center],
      costing,
      contours,
      polygons: true,
      denoise: 1,
      generalize: 50,
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Valhalla isochrone failed (${res.status}): ${text || "unknown error"}`);
  }

  const json = await res.json();
  const features = Array.isArray(json?.features) ? json.features : [];

  const normalized = [];
  for (const f of features) {
    const cutoffMin = Number(f?.properties?.contour);
    const cutoffSec = Number.isFinite(cutoffMin) ? cutoffMin * 60 : Number.NaN;
    if (!f?.geometry) continue;
    normalized.push({
      type: "Feature",
      geometry: f.geometry,
      properties: {
        ...(f.properties || {}),
        cutoffSec,
      },
    });
  }

  normalized.sort((a, b) => Number(a.properties.cutoffSec) - Number(b.properties.cutoffSec));

  return {
    type: "FeatureCollection",
    features: normalized,
  };
}

