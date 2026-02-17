import type { GeocodeResult, IsochroneResponse, LatLon, Mode, WarpGridResponse } from "./types";

export async function geocode(q: string): Promise<GeocodeResult[]> {
  const url = new URL("/api/geocode", window.location.origin);
  url.searchParams.set("q", q);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Geocode failed (${res.status})`);
  const json = (await res.json()) as { results: GeocodeResult[] };
  return json.results;
}

export async function warpGrid(params: {
  a: LatLon;
  b: LatLon;
  mode: Mode;
  rows: number;
  cols: number;
}): Promise<WarpGridResponse> {
  const res = await fetch("/api/warp-grid", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Warp failed (${res.status}): ${text || "unknown error"}`);
  }
  return (await res.json()) as WarpGridResponse;
}

export async function isochrone(params: {
  center: LatLon;
  mode: Mode;
  cutoffsMin: number[];
}): Promise<IsochroneResponse> {
  const url = new URL("/api/isochrone", window.location.origin);
  url.searchParams.set("mode", params.mode);
  url.searchParams.set("lat", `${params.center.lat}`);
  url.searchParams.set("lon", `${params.center.lon}`);
  url.searchParams.set("cutoffs", params.cutoffsMin.join(","));

  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Isochrone failed (${res.status}): ${text || "unknown error"}`);
  }
  return (await res.json()) as IsochroneResponse;
}
