function getOtpConfig() {
  const baseUrl = process.env.OTP_BASE_URL;
  const routerId = process.env.OTP_ROUTER_ID || "default";
  if (!baseUrl) {
    throw new Error("Transit isochrones require OTP_BASE_URL (OpenTripPlanner)");
  }
  return { baseUrl: baseUrl.replace(/\/+$/, ""), routerId };
}

export async function otpIsochroneFeatureCollection({ center, cutoffsSec }) {
  const { baseUrl } = getOtpConfig();

  const allFeatures = [];
  for (const cutoffSec of cutoffsSec) {
    const url = new URL(`${baseUrl}/otp/traveltime/isochrone`);
    url.searchParams.set("location", `${center.lat},${center.lon}`);
    url.searchParams.set("modes", "WALK,TRANSIT");
    url.searchParams.set("arriveBy", "false");
    url.searchParams.set("time", new Date().toISOString());
    url.searchParams.set("cutoff", formatCutoff(cutoffSec));

    const res = await fetch(url, {
      headers: {
        "User-Agent": process.env.OTP_USER_AGENT || "mapper-distorter/0.1 (prototype)",
      },
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`OTP isochrone failed (${res.status}): ${text || "unknown error"}`);
    }
    const json = await res.json();
    const features = Array.isArray(json?.features) ? json.features : [];
    for (const f of features) {
      if (!f?.geometry) continue;
      allFeatures.push({
        type: "Feature",
        geometry: f.geometry,
        properties: {
          ...(f.properties || {}),
          cutoffSec,
        },
      });
    }
  }

  allFeatures.sort((a, b) => Number(a.properties.cutoffSec) - Number(b.properties.cutoffSec));

  return {
    type: "FeatureCollection",
    features: allFeatures,
  };
}

function formatCutoff(totalSeconds) {
  const s = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const minutes = Math.floor(s / 60);
  const seconds = s % 60;
  return `${minutes}M${seconds}S`;
}
