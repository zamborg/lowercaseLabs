import type maplibregl from "maplibre-gl";
import { useEffect, useMemo } from "react";
import type { GeoJSONFeatureCollection } from "./types";

type Offset = { dLat: number; dLon: number };

type Props = {
  map: maplibregl.Map | null;
  featureCollection: GeoJSONFeatureCollection | null;
  offset: Offset;
  visible?: boolean;
};

const SOURCE_ID = "isochrone-src";
const FILL_LAYER_ID = "isochrone-fill";
const LINE_LAYER_ID = "isochrone-line";

export function IsochroneOverlay({ map, featureCollection, offset, visible = true }: Props) {
  const translated = useMemo(() => {
    if (!featureCollection) return null;
    return translateFeatureCollection(featureCollection, offset);
  }, [featureCollection, offset]);

  const cutoffRange = useMemo(() => {
    if (!featureCollection?.features?.length) return null;
    const values = featureCollection.features
      .map((f) => Number(f.properties?.cutoffSec))
      .filter((n) => Number.isFinite(n));
    if (!values.length) return null;
    return { min: Math.min(...values), max: Math.max(...values) };
  }, [featureCollection]);

  useEffect(() => {
    if (!map) return;
    const ensureAdded = () => {
      if (map.getSource(SOURCE_ID)) return;

      map.addSource(SOURCE_ID, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });

      map.addLayer({
        id: FILL_LAYER_ID,
        type: "fill",
        source: SOURCE_ID,
        paint: {
          "fill-color": "rgba(102, 227, 255, 0.16)",
          "fill-opacity": 1,
        },
      });

      map.addLayer({
        id: LINE_LAYER_ID,
        type: "line",
        source: SOURCE_ID,
        paint: {
          "line-color": "rgba(102, 227, 255, 0.85)",
          "line-width": 2,
          "line-opacity": 0.85,
        },
      });
    };

    if (map.isStyleLoaded()) ensureAdded();
    else map.once("load", ensureAdded);

    return () => {
      if (!map.getStyle()) return;
      if (map.getLayer(FILL_LAYER_ID)) map.removeLayer(FILL_LAYER_ID);
      if (map.getLayer(LINE_LAYER_ID)) map.removeLayer(LINE_LAYER_ID);
      if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID);
    };
  }, [map]);

  useEffect(() => {
    if (!map) return;
    const src = map.getSource(SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
    if (!src) return;

    const data = translated && visible ? translated : { type: "FeatureCollection", features: [] };
    src.setData(data);
  }, [map, translated, visible]);

  useEffect(() => {
    if (!map) return;
    if (!cutoffRange) return;
    if (!map.getLayer(FILL_LAYER_ID)) return;

    const { min, max } = cutoffRange;
    const color = [
      "interpolate",
      ["linear"],
      ["to-number", ["get", "cutoffSec"]],
      min,
      "rgba(102, 227, 255, 0.20)",
      max,
      "rgba(102, 227, 255, 0.06)",
    ];
    map.setPaintProperty(FILL_LAYER_ID, "fill-color", color);
  }, [map, cutoffRange]);

  return null;
}

function translateFeatureCollection(featureCollection: GeoJSONFeatureCollection, offset: Offset) {
  return {
    type: "FeatureCollection",
    features: featureCollection.features.map((f) => ({
      ...f,
      geometry: translateGeometry(f.geometry, offset),
    })),
  } satisfies GeoJSONFeatureCollection;
}

function translateGeometry(
  geometry: GeoJSONFeatureCollection["features"][number]["geometry"],
  offset: Offset,
) {
  if (geometry.type === "Polygon") {
    return {
      type: "Polygon",
      coordinates: geometry.coordinates.map((ring) => translateRing(ring, offset)),
    };
  }
  return {
    type: "MultiPolygon",
    coordinates: geometry.coordinates.map((poly) => poly.map((ring) => translateRing(ring, offset))),
  };
}

function translateRing(ring: number[][], offset: Offset) {
  return ring.map(([lon, lat]) => [lon + offset.dLon, lat + offset.dLat]);
}
