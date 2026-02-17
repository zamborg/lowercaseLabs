import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { geocode, isochrone, warpGrid } from "./api";
import { IsochroneOverlay } from "./IsochroneOverlay";
import type { GeocodeResult, IsochroneResponse, LatLon, Mode, WarpGridResponse } from "./types";
import { Overlay } from "./Overlay";

const DEFAULT_HOME: LatLon = { lat: 37.773972, lon: -122.431297 }; // SF-ish
const DEFAULT_WORK: LatLon = { lat: 37.7892, lon: -122.4016 }; // downtown-ish

export function App() {
  const mapDivRef = useRef<HTMLDivElement | null>(null);
  const [map, setMap] = useState<maplibregl.Map | null>(null);
  const [tab, setTab] = useState<"warp" | "isochrone">("warp");
  const [mode, setMode] = useState<Mode>("auto");
  const [home, setHome] = useState<LatLon | null>(DEFAULT_HOME);
  const [work, setWork] = useState<LatLon | null>(DEFAULT_WORK);
  const [homeQuery, setHomeQuery] = useState("San Francisco, CA");
  const [workQuery, setWorkQuery] = useState("Ferry Building, San Francisco");
  const [homeHits, setHomeHits] = useState<GeocodeResult[]>([]);
  const [workHits, setWorkHits] = useState<GeocodeResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warp, setWarp] = useState<WarpGridResponse | null>(null);
  const [showOriginalGrid, setShowOriginalGrid] = useState(false);
  const [clickMode, setClickMode] = useState<"home" | "work" | null>(null);
  const [iso, setIso] = useState<IsochroneResponse | null>(null);
  const [isoCutoffsText, setIsoCutoffsText] = useState("15, 30, 45");
  const [isoOffset, setIsoOffset] = useState<{ dLat: number; dLon: number }>({ dLat: 0, dLon: 0 });
  const [isoDrag, setIsoDrag] = useState(false);
  const isoOffsetRef = useRef(isoOffset);

  useEffect(() => {
    isoOffsetRef.current = isoOffset;
  }, [isoOffset]);

  useEffect(() => {
    if (!mapDivRef.current || map) return;
    const m = new maplibregl.Map({
      container: mapDivRef.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
          },
        },
        layers: [
          {
            id: "osm",
            type: "raster",
            source: "osm",
          },
        ],
      },
      center: [DEFAULT_HOME.lon, DEFAULT_HOME.lat],
      zoom: 12.2,
    });

    m.addControl(new maplibregl.NavigationControl(), "top-right");
    setMap(m);

    return () => {
      m.remove();
    };
  }, [map]);

  useEffect(() => {
    if (!map) return;
    const onClick = (e: maplibregl.MapMouseEvent & maplibregl.EventData) => {
      if (!clickMode) return;
      const ll = { lat: e.lngLat.lat, lon: e.lngLat.lng };
      if (clickMode === "home") {
        setHome(ll);
        setHomeHits([]);
        setHomeQuery(`${ll.lat.toFixed(5)}, ${ll.lon.toFixed(5)}`);
      } else {
        setWork(ll);
        setWorkHits([]);
        setWorkQuery(`${ll.lat.toFixed(5)}, ${ll.lon.toFixed(5)}`);
      }
      setClickMode(null);
    };
    map.on("click", onClick);
    return () => {
      map.off("click", onClick);
    };
  }, [map, clickMode]);

  useEffect(() => {
    if (!map) return;
    if (!isoDrag) {
      map.dragPan.enable();
      map.getCanvas().style.cursor = "";
      return;
    }
    map.dragPan.disable();
    map.getCanvas().style.cursor = "grab";

    let dragging = false;
    let start: maplibregl.LngLat | null = null;
    let startOffset = { dLat: 0, dLon: 0 };

    const onMouseDown = (e: maplibregl.MapMouseEvent & maplibregl.EventData) => {
      if (!iso) return;
      dragging = true;
      start = e.lngLat;
      startOffset = isoOffsetRef.current;
      map.getCanvas().style.cursor = "grabbing";
    };

    const onMouseMove = (e: maplibregl.MapMouseEvent & maplibregl.EventData) => {
      if (!dragging || !start) return;
      const dLon = e.lngLat.lng - start.lng;
      const dLat = e.lngLat.lat - start.lat;
      setIsoOffset({ dLon: startOffset.dLon + dLon, dLat: startOffset.dLat + dLat });
    };

    const onMouseUp = () => {
      dragging = false;
      start = null;
      map.getCanvas().style.cursor = "grab";
    };

    map.on("mousedown", onMouseDown);
    map.on("mousemove", onMouseMove);
    map.on("mouseup", onMouseUp);
    map.on("mouseout", onMouseUp);

    return () => {
      map.off("mousedown", onMouseDown);
      map.off("mousemove", onMouseMove);
      map.off("mouseup", onMouseUp);
      map.off("mouseout", onMouseUp);
      map.dragPan.enable();
      map.getCanvas().style.cursor = "";
    };
  }, [map, isoDrag, iso]);

  const statusBadge = useMemo(() => {
    if (!warp) return null;
    const minutes = Math.round(warp.meta.timeABSec / 60);
    const km = (warp.meta.distanceABMeters / 1000).toFixed(1);
    return (
      <span className="badge">
        <strong>{minutes} min</strong> <span>({km} km, {warp.meta.mode})</span>
      </span>
    );
  }, [warp]);

  const runWarp = async () => {
    setError(null);
    if (!home || !work) {
      setError("Set both Home and Work first.");
      return;
    }
    setBusy(true);
    try {
      const w = await warpGrid({ a: home, b: work, mode, rows: 17, cols: 17 });
      setWarp(w);
      if (map) {
        map.fitBounds(
          [
            [w.meta.bbox.minLon, w.meta.bbox.minLat],
            [w.meta.bbox.maxLon, w.meta.bbox.maxLat],
          ],
          { padding: 40, duration: 450 },
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Warp failed");
    } finally {
      setBusy(false);
    }
  };

  const runIsochrone = async () => {
    setError(null);
    if (!home) {
      setError("Set Home first (isochrones are centered on Home).");
      return;
    }
    const cutoffsMin = isoCutoffsText
      .split(",")
      .map((s) => Number(s.trim()))
      .filter((n) => Number.isFinite(n) && n > 0);
    if (!cutoffsMin.length) {
      setError("Enter cutoffs like: 15, 30, 45");
      return;
    }

    setBusy(true);
    try {
      const resp = await isochrone({ center: home, mode, cutoffsMin });
      setIso(resp);
      setIsoOffset({ dLat: 0, dLon: 0 });
      if (map) {
        const bbox = bboxFromFeatureCollection(resp.featureCollection);
        if (bbox) map.fitBounds(bbox, { padding: 40, duration: 450 });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Isochrone failed");
    } finally {
      setBusy(false);
    }
  };

  const doGeocode = async (which: "home" | "work") => {
    setError(null);
    const q = which === "home" ? homeQuery : workQuery;
    if (!q.trim()) return;
    setBusy(true);
    try {
      const hits = await geocode(q);
      if (which === "home") setHomeHits(hits);
      else setWorkHits(hits);
      if (hits.length === 1) {
        const ll = { lat: hits[0].lat, lon: hits[0].lon };
        if (which === "home") setHome(ll);
        else setWork(ll);
        if (map) map.flyTo({ center: [ll.lon, ll.lat], zoom: 13, duration: 450 });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Geocode failed");
    } finally {
      setBusy(false);
    }
  };

  const pickHit = (which: "home" | "work", h: GeocodeResult) => {
    const ll = { lat: h.lat, lon: h.lon };
    if (which === "home") {
      setHome(ll);
      setHomeQuery(h.displayName);
      setHomeHits([]);
    } else {
      setWork(ll);
      setWorkQuery(h.displayName);
      setWorkHits([]);
    }
    if (map) map.flyTo({ center: [ll.lon, ll.lat], zoom: 13, duration: 450 });
  };

  const swap = () => {
    setHome(work);
    setWork(home);
    setHomeQuery(workQuery);
    setWorkQuery(homeQuery);
    setHomeHits([]);
    setWorkHits([]);
    setWarp(null);
  };

  return (
    <div className="app">
      <div className="panel">
        <h1 className="title">Mapper</h1>
        {tab === "warp" ? statusBadge : null}

        <div className="pillRow" style={{ marginTop: 0 }}>
          <button className="pill" data-active={tab === "warp"} onClick={() => setTab("warp")} type="button">
            Warp
          </button>
          <button
            className="pill"
            data-active={tab === "isochrone"}
            onClick={() => setTab("isochrone")}
            type="button"
          >
            Isochrone
          </button>
        </div>

        <div className="row">
          <label>Home</label>
          <button
            className="pill"
            data-active={clickMode === "home"}
            onClick={() => setClickMode(clickMode === "home" ? null : "home")}
            type="button"
          >
            Click to set
          </button>
          <input
            className="input"
            value={homeQuery}
            onChange={(e) => setHomeQuery(e.target.value)}
            placeholder="Search home address…"
            onKeyDown={(e) => {
              if (e.key === "Enter") doGeocode("home");
            }}
          />
        </div>
        <div className="pillRow">
          <button
            className="btn"
            type="button"
            onClick={() => doGeocode("home")}
            disabled={busy}
          >
            Search
          </button>
          <button
            className="btn"
            type="button"
            onClick={() => {
              setHome(DEFAULT_HOME);
              setHomeQuery("San Francisco, CA");
              setHomeHits([]);
              setWarp(null);
            }}
            disabled={busy}
          >
            Reset
          </button>
        </div>
        {homeHits.length > 1 ? (
          <div className="hint">
            {homeHits.slice(0, 5).map((h) => (
              <div key={`${h.lat},${h.lon}`}>
                <button className="btn" type="button" onClick={() => pickHit("home", h)}>
                  {h.displayName}
                </button>
              </div>
            ))}
          </div>
        ) : null}

        <div className="row">
          <label>Work</label>
          <button
            className="pill"
            data-active={clickMode === "work"}
            onClick={() => setClickMode(clickMode === "work" ? null : "work")}
            type="button"
          >
            Click to set
          </button>
          <input
            className="input"
            value={workQuery}
            onChange={(e) => setWorkQuery(e.target.value)}
            placeholder="Search work address…"
            onKeyDown={(e) => {
              if (e.key === "Enter") doGeocode("work");
            }}
          />
        </div>
        <div className="pillRow">
          <button
            className="btn"
            type="button"
            onClick={() => doGeocode("work")}
            disabled={busy}
          >
            Search
          </button>
          <button className="btn" type="button" onClick={swap} disabled={busy}>
            Swap
          </button>
        </div>
        {workHits.length > 1 ? (
          <div className="hint">
            {workHits.slice(0, 5).map((h) => (
              <div key={`${h.lat},${h.lon}`}>
                <button className="btn" type="button" onClick={() => pickHit("work", h)}>
                  {h.displayName}
                </button>
              </div>
            ))}
          </div>
        ) : null}

        <div className="hint" style={{ marginTop: 10 }}>
          Mode
        </div>
        <div className="pillRow">
          {(["auto", "bicycle", "pedestrian", "transit"] as const).map((m) => (
            <button
              key={m}
              className="pill"
              data-active={mode === m}
              onClick={() => setMode(m)}
              type="button"
            >
              {m}
            </button>
          ))}
        </div>

        {tab === "warp" ? (
          <>
            <div className="buttonRow">
              <button className="btn btnPrimary" type="button" onClick={runWarp} disabled={busy}>
                {busy ? "Working…" : "Distort"}
              </button>
              <button
                className="btn"
                type="button"
                onClick={() => setShowOriginalGrid((v) => !v)}
                disabled={!warp}
              >
                {showOriginalGrid ? "Hide original" : "Show original"}
              </button>
            </div>
            <p className="hint">
              Distortion uses travel-time “distance” with two anchors (Home/Work). The basemap stays normal; the cyan
              grid shows the warped geometry.
            </p>
          </>
        ) : (
          <>
            <div className="row">
              <label>Cutoffs (min)</label>
              <input
                className="input"
                value={isoCutoffsText}
                onChange={(e) => setIsoCutoffsText(e.target.value)}
                placeholder="15, 30, 45"
              />
            </div>
            <div className="buttonRow">
              <button className="btn btnPrimary" type="button" onClick={runIsochrone} disabled={busy}>
                {busy ? "Working…" : "Generate"}
              </button>
              <button
                className="btn"
                type="button"
                onClick={() => {
                  setIsoOffset({ dLat: 0, dLon: 0 });
                  setIsoDrag(false);
                }}
                disabled={!iso}
              >
                Reset overlay
              </button>
            </div>
            <div className="pillRow">
              <button className="pill" data-active={isoDrag} onClick={() => setIsoDrag((v) => !v)} type="button" disabled={!iso}>
                {isoDrag ? "Dragging overlay" : "Drag overlay"}
              </button>
            </div>
            <p className="hint">
              Generates travel-time isochrones centered on Home. Turn on “Drag overlay” to superimpose the shape
              somewhere else (like thetruesize.com).
            </p>
          </>
        )}
        {error ? <p className="error">{error}</p> : null}
      </div>

      <div className="mapWrap">
        <div className="map" ref={mapDivRef} />
        <Overlay map={map} warp={warp} showOriginal={showOriginalGrid} />
        <IsochroneOverlay
          map={map}
          featureCollection={iso?.featureCollection ?? null}
          offset={isoOffset}
          visible={!!iso}
        />
      </div>
    </div>
  );
}

function bboxFromFeatureCollection(featureCollection: IsochroneResponse["featureCollection"]) {
  let minLon = Number.POSITIVE_INFINITY;
  let minLat = Number.POSITIVE_INFINITY;
  let maxLon = Number.NEGATIVE_INFINITY;
  let maxLat = Number.NEGATIVE_INFINITY;

  for (const f of featureCollection.features) {
    const g = f.geometry;
    const polys = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
    for (const poly of polys) {
      for (const ring of poly) {
        for (const [lon, lat] of ring) {
          if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
          minLon = Math.min(minLon, lon);
          minLat = Math.min(minLat, lat);
          maxLon = Math.max(maxLon, lon);
          maxLat = Math.max(maxLat, lat);
        }
      }
    }
  }

  if (![minLon, minLat, maxLon, maxLat].every(Number.isFinite)) return null;
  return [
    [minLon, minLat],
    [maxLon, maxLat],
  ] as [maplibregl.LngLatLike, maplibregl.LngLatLike];
}
