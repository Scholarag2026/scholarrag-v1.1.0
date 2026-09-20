"use client";

import { useEffect, useRef } from "react";
import type { GraphNode } from "@/hooks/use-graph";
import { type ColorMode, VIRIDIS, TABLEAU_10 } from "@/components/graph-legend";

interface MinimapProps {
  nodes: Array<{ x: number; y: number; data: GraphNode }>;
  focusNodeId: string | null;
  viewport: { x: number; y: number; width: number; height: number } | null;
  onNavigate: (x: number, y: number) => void;
  colorMode: ColorMode;
  yearRange: [number, number];
}

function getMinimapColor(
  node: GraphNode,
  isFocus: boolean,
  colorMode: ColorMode,
  yearRange: [number, number],
): string {
  if (isFocus) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue("--ds-primary").trim() || "#0D5D56";
  }

  if (colorMode === "year") {
    if (node.year == null) return "#8B7D6B";
    const [minY, maxY] = yearRange;
    const t = maxY === minY ? 0.5 : (node.year - minY) / (maxY - minY);
    const idx = Math.round(t * (VIRIDIS.length - 1));
    return VIRIDIS[Math.max(0, Math.min(VIRIDIS.length - 1, idx))];
  }

  if (colorMode === "cluster") {
    if (node.cluster_id == null) return "#8B7D6B";
    return TABLEAU_10[node.cluster_id % TABLEAU_10.length];
  }

  // quality mode
  if (node.in_library) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue("--ds-success").trim() || "#059669";
  }
  return "#6b7280";
}

const MINIMAP_W = 160;
const MINIMAP_H = 110;

export function GraphMinimap({ nodes, focusNodeId, viewport, onNavigate, colorMode, yearRange }: MinimapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Compute bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of nodes) {
      if (n.x < minX) minX = n.x;
      if (n.x > maxX) maxX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.y > maxY) maxY = n.y;
    }
    const pad = 20;
    const rangeX = (maxX - minX) || 1;
    const rangeY = (maxY - minY) || 1;
    const scale = Math.min((MINIMAP_W - pad * 2) / rangeX, (MINIMAP_H - pad * 2) / rangeY);
    const toX = (x: number) => pad + (x - minX) * scale;
    const toY = (y: number) => pad + (y - minY) * scale;

    ctx.clearRect(0, 0, MINIMAP_W, MINIMAP_H);

    // Resolve CSS custom properties for canvas rendering
    const styles = getComputedStyle(canvas);
    const colorBorder = styles.getPropertyValue("--ds-border").trim() || "#e2e8f0";

    // Draw nodes
    for (const n of nodes) {
      const isFocus = n.data.id === focusNodeId;
      ctx.beginPath();
      ctx.arc(toX(n.x), toY(n.y), isFocus ? 4 : 2, 0, Math.PI * 2);
      ctx.fillStyle = getMinimapColor(n.data, isFocus, colorMode, yearRange);
      ctx.fill();
    }

    // Draw viewport rectangle
    if (viewport) {
      ctx.strokeStyle = colorBorder;
      ctx.lineWidth = 1;
      ctx.strokeRect(
        toX(viewport.x), toY(viewport.y),
        viewport.width * scale, viewport.height * scale
      );
    }
  }, [nodes, focusNodeId, viewport, colorMode, yearRange]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (nodes.length === 0) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of nodes) {
      if (n.x < minX) minX = n.x;
      if (n.x > maxX) maxX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.y > maxY) maxY = n.y;
    }
    const pad = 20;
    const rangeX = (maxX - minX) || 1;
    const rangeY = (maxY - minY) || 1;
    const scale = Math.min((MINIMAP_W - pad * 2) / rangeX, (MINIMAP_H - pad * 2) / rangeY);
    const graphX = (clickX - pad) / scale + minX;
    const graphY = (clickY - pad) / scale + minY;
    onNavigate(graphX, graphY);
  };

  return (
    <div className="absolute bottom-3 right-3 rounded-lg bg-[var(--ds-bg-card)]/80 border border-[var(--ds-border)] p-1 z-10">
      <canvas
        ref={canvasRef}
        width={MINIMAP_W}
        height={MINIMAP_H}
        className="cursor-pointer"
        onClick={handleClick}
      />
    </div>
  );
}
