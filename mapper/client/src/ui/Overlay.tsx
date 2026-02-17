import type maplibregl from "maplibre-gl";
import React, { useEffect, useMemo, useRef } from "react";
import type { LatLon, WarpGridResponse } from "./types";

type Props = {
  map: maplibregl.Map | null;
  warp: WarpGridResponse | null;
  showOriginal?: boolean;
};

export function Overlay({ map, warp, showOriginal = false }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  const gridLines = useMemo(() => {
    if (!warp) return null;
    const { rows, cols } = warp.grid;
    const idx = (r: number, c: number) => r * cols + c;
    const warped = warp.warped;
    const original = warp.original;
    const lines: { pts: LatLon[]; kind: "warped" | "original" }[] = [];

    for (let r = 0; r < rows; r++) {
      const wPts: LatLon[] = [];
      const oPts: LatLon[] = [];
      for (let c = 0; c < cols; c++) {
        wPts.push(warped[idx(r, c)]);
        oPts.push(original[idx(r, c)]);
      }
      lines.push({ pts: wPts, kind: "warped" });
      if (showOriginal) lines.push({ pts: oPts, kind: "original" });
    }

    for (let c = 0; c < cols; c++) {
      const wPts: LatLon[] = [];
      const oPts: LatLon[] = [];
      for (let r = 0; r < rows; r++) {
        wPts.push(warped[idx(r, c)]);
        oPts.push(original[idx(r, c)]);
      }
      lines.push({ pts: wPts, kind: "warped" });
      if (showOriginal) lines.push({ pts: oPts, kind: "original" });
    }

    return { lines };
  }, [warp, showOriginal]);

  useEffect(() => {
    if (!map) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const { width, height } = parent.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    };

    const draw = () => {
      rafRef.current = null;
      resize();
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      ctx.clearRect(0, 0, w, h);

      if (!warp || !gridLines) return;

      const drawLine = (pts: LatLon[], color: string, width: number) => {
        ctx.beginPath();
        for (let i = 0; i < pts.length; i++) {
          const p = map.project([pts[i].lon, pts[i].lat]);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.stroke();
      };

      for (const line of gridLines.lines) {
        if (line.kind === "original") {
          drawLine(line.pts, "rgba(234, 240, 255, 0.12)", 1);
        } else {
          drawLine(line.pts, "rgba(102, 227, 255, 0.42)", 1.5);
        }
      }

      const drawPin = (ll: LatLon, fill: string) => {
        const p = map.project([ll.lon, ll.lat]);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(0,0,0,0.35)";
        ctx.stroke();
      };

      drawPin(warp.a, "rgba(255, 204, 102, 0.95)");
      drawPin(warp.b, "rgba(255, 107, 107, 0.95)");
    };

    const scheduleDraw = () => {
      if (rafRef.current != null) return;
      rafRef.current = requestAnimationFrame(draw);
    };

    scheduleDraw();
    map.on("render", scheduleDraw);
    window.addEventListener("resize", scheduleDraw);

    return () => {
      map.off("render", scheduleDraw);
      window.removeEventListener("resize", scheduleDraw);
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [map, warp, gridLines]);

  return <canvas className="overlayCanvas" ref={canvasRef} />;
}

