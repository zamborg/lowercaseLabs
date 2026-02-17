export type Mode = "auto" | "bicycle" | "pedestrian" | "transit";

export type LatLon = { lat: number; lon: number };

export type GeoJSONPolygon = {
  type: "Polygon";
  coordinates: number[][][];
};

export type GeoJSONMultiPolygon = {
  type: "MultiPolygon";
  coordinates: number[][][][];
};

export type GeoJSONGeometry = GeoJSONPolygon | GeoJSONMultiPolygon;

export type GeoJSONFeature = {
  type: "Feature";
  geometry: GeoJSONGeometry;
  properties: Record<string, unknown> & { cutoffSec?: number };
};

export type GeoJSONFeatureCollection = {
  type: "FeatureCollection";
  features: GeoJSONFeature[];
};

export type WarpGridResponse = {
  grid: { rows: number; cols: number };
  original: LatLon[];
  warped: LatLon[];
  a: LatLon;
  b: LatLon;
  meta: {
    mode: Mode;
    timeABSec: number;
    distanceABMeters: number;
    metersPerSecond: number;
    bbox: { minLat: number; minLon: number; maxLat: number; maxLon: number };
  };
};

export type GeocodeResult = {
  displayName: string;
  lat: number;
  lon: number;
};

export type IsochroneResponse = {
  center: LatLon;
  mode: Mode;
  featureCollection: GeoJSONFeatureCollection;
};
