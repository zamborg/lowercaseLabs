import { clamp, latLonFromMercator, mercatorFromLatLon } from "./geo.mjs";

function circleIntersections(a, r0, b, r1) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const d = Math.hypot(dx, dy);
  if (d === 0) return null;
  if (d > r0 + r1) return null;
  if (d < Math.abs(r0 - r1)) return null;

  const aDist = (r0 * r0 - r1 * r1 + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(0, r0 * r0 - aDist * aDist));
  const xm = a.x + (aDist * dx) / d;
  const ym = a.y + (aDist * dy) / d;

  const rx = (-dy * (h / d));
  const ry = (dx * (h / d));
  const p1 = { x: xm + rx, y: ym + ry };
  const p2 = { x: xm - rx, y: ym - ry };
  return { p1, p2 };
}

function sideSign(a, b, p) {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const wx = p.x - a.x;
  const wy = p.y - a.y;
  const cross = vx * wy - vy * wx;
  return cross === 0 ? 0 : cross > 0 ? 1 : -1;
}

function adjustRadiiToTriangleInequality(rA, rB, dAB) {
  const minSum = dAB + 1e-6;
  const maxDiff = dAB - 1e-6;

  let a = rA;
  let b = rB;

  if (a + b < minSum) {
    const deficit = minSum - (a + b);
    a += deficit / 2;
    b += deficit / 2;
  }
  const diff = Math.abs(a - b);
  if (diff > maxDiff) {
    const shrink = (diff - maxDiff) / 2;
    if (a > b) a -= shrink;
    else b -= shrink;
  }
  a = Math.max(0.01, a);
  b = Math.max(0.01, b);
  return { rA: a, rB: b };
}

export function warpGridTwoAnchors({
  aLatLon,
  bLatLon,
  originalPointsLatLon,
  secondsFromA,
  secondsFromB,
  metersPerSecond,
}) {
  const a = mercatorFromLatLon(aLatLon);
  const b = mercatorFromLatLon(bLatLon);
  const dAB = Math.hypot(b.x - a.x, b.y - a.y);

  const warped = [];

  for (let i = 0; i < originalPointsLatLon.length; i++) {
    const p0 = mercatorFromLatLon(originalPointsLatLon[i]);
    const sA = secondsFromA[i];
    const sB = secondsFromB[i];

    if (!Number.isFinite(sA) || !Number.isFinite(sB)) {
      warped.push(originalPointsLatLon[i]);
      continue;
    }

    let rA = clamp(sA, 0, 24 * 3600) * metersPerSecond;
    let rB = clamp(sB, 0, 24 * 3600) * metersPerSecond;

    const adj = adjustRadiiToTriangleInequality(rA, rB, dAB);
    rA = adj.rA;
    rB = adj.rB;

    const inter = circleIntersections(a, rA, b, rB);
    if (!inter) {
      warped.push(originalPointsLatLon[i]);
      continue;
    }

    const origSign = sideSign(a, b, p0);
    const s1 = sideSign(a, b, inter.p1);
    const s2 = sideSign(a, b, inter.p2);

    let chosen = inter.p1;
    if (origSign !== 0) {
      if (s1 === origSign && s2 !== origSign) chosen = inter.p1;
      else if (s2 === origSign && s1 !== origSign) chosen = inter.p2;
      else {
        const d1 = Math.hypot(inter.p1.x - p0.x, inter.p1.y - p0.y);
        const d2 = Math.hypot(inter.p2.x - p0.x, inter.p2.y - p0.y);
        chosen = d1 <= d2 ? inter.p1 : inter.p2;
      }
    } else {
      const d1 = Math.hypot(inter.p1.x - p0.x, inter.p1.y - p0.y);
      const d2 = Math.hypot(inter.p2.x - p0.x, inter.p2.y - p0.y);
      chosen = d1 <= d2 ? inter.p1 : inter.p2;
    }

    warped.push(latLonFromMercator(chosen));
  }

  return warped;
}

