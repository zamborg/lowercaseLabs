import { valhallaIsochroneFeatureCollection } from "./valhalla.mjs";
import { otpIsochroneFeatureCollection } from "./otp.mjs";
import { SUPPORTED_MODES as VALHALLA_MODES } from "../valhalla.mjs";

export async function getIsochroneFeatureCollection({ center, mode, cutoffsSec }) {
  if (!center || !Number.isFinite(center.lat) || !Number.isFinite(center.lon)) {
    throw new Error("Invalid center");
  }
  if (!Array.isArray(cutoffsSec) || cutoffsSec.length === 0) {
    throw new Error("Missing cutoffsSec");
  }

  const normalizedCutoffsSec = Array.from(new Set(cutoffsSec))
    .map((n) => Number(n))
    .filter((n) => Number.isFinite(n) && n > 0)
    .sort((a, b) => a - b);

  if (!normalizedCutoffsSec.length) throw new Error("Invalid cutoffsSec");

  if (VALHALLA_MODES.has(mode)) {
    return await valhallaIsochroneFeatureCollection({
      center,
      costing: mode,
      cutoffsSec: normalizedCutoffsSec,
    });
  }

  if (mode === "transit") {
    return await otpIsochroneFeatureCollection({
      center,
      cutoffsSec: normalizedCutoffsSec,
    });
  }

  throw new Error("Unsupported mode");
}

