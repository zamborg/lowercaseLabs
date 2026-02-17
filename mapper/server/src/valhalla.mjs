const VALHALLA_BASE = process.env.VALHALLA_BASE_URL || "https://valhalla1.openstreetmap.de";

export const SUPPORTED_MODES = new Set(["auto", "bicycle", "pedestrian"]);

export async function sourcesToTargets({ sources, targets, costing }) {
  const res = await fetch(`${VALHALLA_BASE}/sources_to_targets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sources, targets, costing }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Valhalla sources_to_targets failed (${res.status}): ${text || "unknown error"}`);
  }
  return await res.json();
}

