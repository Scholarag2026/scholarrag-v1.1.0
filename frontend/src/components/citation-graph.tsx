"use client";

import { GraphSidePanel } from "@/components/graph-side-panel";
import { VIRIDIS, TABLEAU_10 } from "@/components/graph-legend";
import type { GraphData, GraphNode } from "@/hooks/use-graph";
import { useExpandNode } from "@/hooks/use-graph";
import * as d3 from "d3";
import { useTranslations } from "next-intl";
import { toast } from "sonner";
import { useCallback, useEffect, useRef, useState } from "react";

interface CitationGraphProps {
  data: GraphData;
  projectId: string;
  onAddToLibrary: (node: GraphNode) => void;
  focusNodeId: string | null;
  onFocusChange: (nodeId: string, nodeTitle: string) => void;
  onPositionsUpdate?: (nodes: Array<{ x: number; y: number; data: GraphNode }>) => void;
  onRegisterControls?: (controls: { fitAll: () => void; centerOnFocus: () => void; navigateTo: (x: number, y: number) => void }) => void;
  colorMode: "year" | "cluster" | "quality";
  searchQuery: string;
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  data: GraphNode;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  source: SimNode | string;
  target: SimNode | string;
}

function nodeRadius(node: GraphNode, isFocus: boolean): number {
  if (isFocus) return 22;
  if (!node.in_library) return 8;
  const count = node.citation_count ?? 0;
  return Math.max(8, Math.min(30, 8 + Math.sqrt(count) * 1.5));
}

function nodeColor(
  node: GraphNode,
  isFocus: boolean,
  colorMode: "year" | "cluster" | "quality",
  yearDomain: [number, number],
): string {
  if (isFocus) return "var(--ds-primary)";

  if (colorMode === "year") {
    if (node.year == null) return "var(--ds-text-muted)";
    const [minY, maxY] = yearDomain;
    const t = maxY === minY ? 0.5 : (node.year - minY) / (maxY - minY);
    const idx = Math.round(t * (VIRIDIS.length - 1));
    return VIRIDIS[Math.max(0, Math.min(VIRIDIS.length - 1, idx))];
  }

  if (colorMode === "cluster") {
    if (node.cluster_id == null) return "var(--ds-text-muted)";
    return TABLEAU_10[node.cluster_id % TABLEAU_10.length];
  }

  // quality mode — preserved from original
  if (!node.in_library) return "var(--ds-text-secondary)";
  if (node.quality_score == null) return "var(--ds-text-muted)";
  if (node.quality_score > 0.6) return "var(--ds-success)";
  if (node.quality_score >= 0.3) return "var(--ds-warning)";
  return "var(--ds-error)";
}

export function CitationGraph({
  data,
  projectId,
  onAddToLibrary,
  focusNodeId,
  onFocusChange,
  onPositionsUpdate,
  onRegisterControls,
  colorMode,
  searchQuery,
}: CitationGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);
  const positionsRef = useRef<Map<string, { x: number; y: number }>>(
    new Map(),
  );
  const svgGRef = useRef<d3.Selection<
    SVGGElement,
    unknown,
    null,
    undefined
  > | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const linksGroupRef = useRef<d3.Selection<
    SVGGElement,
    unknown,
    null,
    undefined
  > | null>(null);
  const glowGroupRef = useRef<d3.Selection<
    SVGGElement,
    unknown,
    null,
    undefined
  > | null>(null);
  const nodesGroupRef = useRef<d3.Selection<
    SVGGElement,
    unknown,
    null,
    undefined
  > | null>(null);
  const labelsGroupRef = useRef<d3.Selection<
    SVGGElement,
    unknown,
    null,
    undefined
  > | null>(null);
  const initedRef = useRef(false);

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const expandNode = useExpandNode();
  const t = useTranslations("graph");

  // Keep stable refs for callbacks used in D3 handlers
  const setSelectedNodeRef = useRef(setSelectedNode);
  setSelectedNodeRef.current = setSelectedNode;
  const focusNodeIdRef = useRef(focusNodeId);
  focusNodeIdRef.current = focusNodeId;
  // Read inside the data-sync effect so colour changes do not re-run it (see below).
  const colorModeRef = useRef(colorMode);
  colorModeRef.current = colorMode;
  // Tracks which colorMode has actually been painted, so the repaint effect below is a
  // no-op on data/focus churn. colorModeRef is reassigned during render (see above), so
  // it is never the "previous" value — this ref must be separate.
  const paintedColorModeRef = useRef<typeof colorMode | null>(null);

  const handleExpand = useCallback(
    (paperId: string) => {
      expandNode.mutate(
        { projectId, paperId },
        {
          // Without this the spinner just stops on a 502/504 (issue EXPAND-NODE-504).
          onError: (err) => toast.error(err.message || t("expandFailed")),
        },
      );
      const node = data.nodes.find((n) => n.id === paperId);
      onFocusChange(paperId, node?.title ?? "Unknown");
    },
    [expandNode, projectId, onFocusChange, data.nodes, t],
  );

  // ──────────────────────────────────────────────────────────
  // Init effect (mount-only): create SVG structure, zoom, etc.
  // ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    // Clear anything from a potential previous mount
    svg.selectAll("*").remove();

    const g = svg.append("g");
    svgGRef.current = g;

    // Sub-groups in draw order
    linksGroupRef.current = g.append("g").attr("class", "links-group");
    glowGroupRef.current = g.append("g").attr("class", "glow-group");
    nodesGroupRef.current = g.append("g").attr("class", "nodes-group");
    labelsGroupRef.current = g.append("g").attr("class", "labels-group");

    // Zoom
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
        const k = event.transform.k;
        if (labelsGroupRef.current) {
          labelsGroupRef.current.selectAll<SVGTextElement, SimNode>("text")
            .attr("opacity", (d) => {
              if (d.id === focusNodeIdRef.current) return 1;
              if (k < 0.6) return 0;
              if (k < 1.0) return d.data.in_library ? 1 : 0;
              return 1;
            });
        }
      });
    svg.call(zoom);
    zoomRef.current = zoom;

    // Register external controls
    onRegisterControls?.({
      fitAll: () => {
        d3.select(svgRef.current!).transition().duration(500).call(
          zoom.transform, d3.zoomIdentity
        );
      },
      centerOnFocus: () => {
        const focusId = focusNodeIdRef.current;
        if (!focusId || !simulationRef.current) return;
        const node = simulationRef.current.nodes().find((n) => n.id === focusId);
        if (node && node.x != null && node.y != null) {
          const w = container.clientWidth;
          const h = container.clientHeight;
          d3.select(svgRef.current!).transition().duration(500).call(
            zoom.transform,
            d3.zoomIdentity.translate(w / 2 - node.x, h / 2 - node.y).scale(1.2)
          );
        }
      },
      navigateTo: (x: number, y: number) => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        d3.select(svgRef.current!).transition().duration(300).call(
          zoom.transform,
          d3.zoomIdentity.translate(w / 2 - x, h / 2 - y).scale(1)
        );
      },
    });

    // Tooltip div
    const tooltipDiv = document.createElement("div");
    tooltipDiv.className =
      "absolute pointer-events-none bg-[var(--ds-bg-card)] border border-[var(--ds-border)] rounded px-2 py-1 text-xs text-[var(--ds-text-body)] opacity-0 transition-opacity z-20 shadow-md";
    container.appendChild(tooltipDiv);
    tooltipRef.current = tooltipDiv;

    // Empty simulation
    const simulation = d3
      .forceSimulation<SimNode, SimLink>([])
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>([])
          .id((d) => d.id)
          .distance(80),
      )
      .force("charge", d3.forceManyBody().strength(-200).distanceMax(300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force(
        "collide",
        d3
          .forceCollide<SimNode>()
          .radius((d) =>
            nodeRadius(d.data, d.id === focusNodeIdRef.current) + 4,
          ),
      )
      .stop();
    simulationRef.current = simulation;

    initedRef.current = true;

    return () => {
      simulation.stop();
      if (tooltipDiv.parentNode) tooltipDiv.parentNode.removeChild(tooltipDiv);
      initedRef.current = false;
    };
  }, []); // mount-only

  // ──────────────────────────────────────────────────────────
  // Data-sync effect: D3 general update pattern on data/focus changes
  // ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (
      !initedRef.current ||
      !svgRef.current ||
      !containerRef.current ||
      !simulationRef.current ||
      !linksGroupRef.current ||
      !glowGroupRef.current ||
      !nodesGroupRef.current ||
      !labelsGroupRef.current
    )
      return;
    if (data.nodes.length === 0) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;
    const simulation = simulationRef.current;
    const svg = d3.select(svgRef.current);
    const zoom = zoomRef.current!;
    const tooltipDiv = tooltipRef.current!;

    // ── Compute year domain for color encoding ──
    const years = data.nodes.map((n) => n.year).filter((y): y is number => y != null);
    const yearDomain: [number, number] = years.length
      ? [Math.min(...years), Math.max(...years)]
      : [2000, 2025];

    // ── Snapshot current positions ──
    const oldNodes = simulation.nodes();
    for (const n of oldNodes) {
      if (n.x != null && n.y != null) {
        positionsRef.current.set(n.id, { x: n.x, y: n.y });
      }
    }

    // ── Build new SimNode / SimLink arrays ──
    const nodes: SimNode[] = data.nodes.map((n) => {
      const saved = positionsRef.current.get(n.id);
      const simNode: SimNode = { id: n.id, data: n };
      if (saved) {
        simNode.x = saved.x;
        simNode.y = saved.y;
      }
      // New nodes without saved positions will be placed below
      return simNode;
    });

    // Deterministic initial layout for first render
    if (positionsRef.current.size === 0) {
      const cx = width / 2;
      const cy = height / 2;
      const libraryNodes = nodes.filter((n) => n.data.in_library);
      const externalNodes = nodes.filter((n) => !n.data.in_library);

      // Sort by id for determinism
      libraryNodes.sort((a, b) => a.id.localeCompare(b.id));
      externalNodes.sort((a, b) => a.id.localeCompare(b.id));

      // Place library in inner circle
      const innerR = Math.min(width, height) * 0.15;
      libraryNodes.forEach((n, i) => {
        const angle = (2 * Math.PI * i) / (libraryNodes.length || 1);
        n.x = cx + innerR * Math.cos(angle);
        n.y = cy + innerR * Math.sin(angle);
      });

      // Place external in outer ring
      const outerR = Math.min(width, height) * 0.35;
      externalNodes.forEach((n, i) => {
        const angle = (2 * Math.PI * i) / (externalNodes.length || 1);
        n.x = cx + outerR * Math.cos(angle);
        n.y = cy + outerR * Math.sin(angle);
      });
    } else {
      // For nodes without saved positions (new nodes from expansion), place near focus
      for (const n of nodes) {
        if (n.x == null) {
          const focusPos = focusNodeId ? positionsRef.current.get(focusNodeId) : null;
          if (focusPos) {
            n.x = focusPos.x + (Math.random() - 0.5) * 80;
            n.y = focusPos.y + (Math.random() - 0.5) * 80;
          } else {
            n.x = width / 2 + (Math.random() - 0.5) * 100;
            n.y = height / 2 + (Math.random() - 0.5) * 100;
          }
        }
      }
    }

    const nodeMap = new Map(nodes.map((n) => [n.id, n]));

    const links: SimLink[] = data.edges
      .filter((e) => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map((e) => ({
        source: e.source,
        target: e.target,
      }));

    // ── Update simulation ──
    simulation.nodes(nodes);
    (
      simulation.force("link") as d3.ForceLink<SimNode, SimLink>
    ).links(links);
    // Update collide radius to reflect current focus
    (
      simulation.force("collide") as d3.ForceCollide<SimNode>
    ).radius((d) => nodeRadius(d.data, d.id === focusNodeId) + 4);

    // For first render, run simulation synchronously to reach stable state
    if (positionsRef.current.size === 0 && nodes.length > 0) {
      simulation.alpha(1);
      for (let i = 0; i < 120; i++) simulation.tick();
      // Save the stabilized positions
      for (const n of nodes) {
        if (n.x != null && n.y != null) {
          positionsRef.current.set(n.id, { x: n.x, y: n.y });
        }
      }
    }

    // ── LINKS: enter / update / exit ──
    const linkSel = linksGroupRef
      .current!.selectAll<SVGLineElement, SimLink>("line")
      .data(links, (d) => {
        const s = typeof d.source === "string" ? d.source : (d.source as SimNode).id;
        const t = typeof d.target === "string" ? d.target : (d.target as SimNode).id;
        return `${s}->${t}`;
      });

    linkSel
      .exit<SimLink>()
      .transition()
      .duration(300)
      .attr("stroke-opacity", 0)
      .remove();

    const linkEnter = linkSel
      .enter()
      .append("line")
      .style("stroke", "#9CA3AF")
      .attr("stroke-width", 1.2)
      .attr("stroke-opacity", 0);

    linkEnter
      .transition()
      .duration(400)
      .attr("stroke-opacity", 0.45);

    const linkMerged = linkEnter.merge(linkSel);
    // Existing links keep their opacity
    linkSel.transition().duration(300).attr("stroke-opacity", 0.45);

    // ── GLOW: enter / update / exit ──
    const focusNodes = nodes.filter((n) => n.id === focusNodeId);

    const glowSel = glowGroupRef
      .current!.selectAll<SVGCircleElement, SimNode>("circle")
      .data(focusNodes, (d) => d.id);

    glowSel
      .exit<SimNode>()
      .transition()
      .duration(300)
      .attr("stroke-opacity", 0)
      .remove();

    const glowEnter = glowSel
      .enter()
      .append("circle")
      .attr("fill", "none")
      .style("stroke", "var(--ds-primary)")
      .attr("stroke-width", 3)
      .attr("stroke-opacity", 0);

    glowEnter
      .transition()
      .duration(400)
      .attr("stroke-opacity", 0.3);

    const glowMerged = glowEnter.merge(glowSel);
    glowMerged.attr("r", (d) => nodeRadius(d.data, true) + 8);
    glowSel.transition().duration(300).attr("stroke-opacity", 0.3);

    // ── NODES: enter / update / exit ──
    const nodeSel = nodesGroupRef
      .current!.selectAll<SVGCircleElement, SimNode>("circle")
      .data(nodes, (d) => d.id);

    nodeSel
      .exit<SimNode>()
      .transition()
      .duration(300)
      .attr("opacity", 0)
      .remove();

    const nodeEnter = nodeSel
      .enter()
      .append("circle")
      .attr("cursor", "pointer")
      .attr("opacity", 0);

    nodeEnter
      .transition()
      .duration(400)
      .attr("opacity", 1);

    const nodeMerged = nodeEnter.merge(nodeSel);

    // Update visual properties for all nodes (enter + update)
    nodeMerged
      .transition()
      .duration(300)
      .attr("r", (d) => nodeRadius(d.data, d.id === focusNodeId))
      .style("fill", (d) =>
        nodeColor(d.data, d.id === focusNodeId, colorModeRef.current, yearDomain),
      )
      .style("stroke", (d) =>
        d.id === focusNodeId
          ? "var(--ds-text-heading)"
          : d.data.in_library
            ? "none"
            : "var(--ds-text-secondary)",
      )
      .attr("stroke-width", (d) =>
        d.id === focusNodeId ? 2.5 : d.data.in_library ? 0 : 1.5,
      )
      .attr("stroke-dasharray", (d) =>
        d.id === focusNodeId
          ? "none"
          : d.data.in_library
            ? "none"
            : "3,3",
      );

    // Click handler on all nodes
    nodeMerged.on("click", (_, d) => {
      setSelectedNodeRef.current(d.data);
    });

    // Drag
    const drag = d3
      .drag<SVGCircleElement, SimNode>()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });
    nodeMerged.call(drag);

    // ── LABELS: enter / update / exit ──
    const labelData = nodes.filter(
      (n) => n.data.in_library || n.id === focusNodeId,
    );

    const labelSel = labelsGroupRef
      .current!.selectAll<SVGTextElement, SimNode>("text")
      .data(labelData, (d) => d.id);

    labelSel
      .exit<SimNode>()
      .transition()
      .duration(300)
      .attr("opacity", 0)
      .remove();

    const labelEnter = labelSel
      .enter()
      .append("text")
      .attr("text-anchor", "middle")
      .attr("opacity", 0);

    labelEnter
      .transition()
      .duration(400)
      .attr("opacity", 1);

    const labelMerged = labelEnter.merge(labelSel);

    labelMerged
      .text((d) => {
        if (d.id === focusNodeId) return d.data.title;
        return d.data.title.length > 30
          ? d.data.title.slice(0, 30) + "..."
          : d.data.title;
      })
      .attr("font-size", (d) => (d.id === focusNodeId ? 11 : 9))
      .attr("font-weight", (d) => (d.id === focusNodeId ? "600" : "normal"))
      .style("fill", (d) =>
        d.id === focusNodeId ? "var(--ds-primary)" : "var(--ds-text-secondary)",
      )
      .attr("dy", (d) => nodeRadius(d.data, d.id === focusNodeId) + 14);

    // Update existing labels' opacity
    labelSel.transition().duration(300).attr("opacity", 1);

    // ── Neighborhood highlighting (hover) ──
    nodeMerged
      .on("mouseenter", (event, d) => {
        // Build connected set
        const connectedIds = new Set<string>();
        connectedIds.add(d.id);
        const currentLinks = (simulation.force("link") as d3.ForceLink<SimNode, SimLink>).links();
        currentLinks.forEach((l) => {
          const s = typeof l.source === "string" ? l.source : (l.source as SimNode).id;
          const t = typeof l.target === "string" ? l.target : (l.target as SimNode).id;
          if (s === d.id) connectedIds.add(t);
          if (t === d.id) connectedIds.add(s);
        });

        // Dim non-connected
        nodeMerged.transition().duration(200)
          .attr("opacity", (n) => connectedIds.has(n.id) ? 1 : 0.15);
        linkMerged.transition().duration(200)
          .attr("stroke-opacity", (l) => {
            const s = (l.source as SimNode).id;
            const t = (l.target as SimNode).id;
            return connectedIds.has(s) && connectedIds.has(t) ? 0.7 : 0.05;
          })
          .style("stroke", (l) => {
            const s = (l.source as SimNode).id;
            const t = (l.target as SimNode).id;
            return connectedIds.has(s) && connectedIds.has(t) ? "var(--ds-primary)" : "#9CA3AF";
          });
        labelMerged.transition().duration(200)
          .attr("opacity", (n) => connectedIds.has(n.id) ? 1 : 0.1);

        // Tooltip
        const authors = d.data.authors.length > 2
          ? `${d.data.authors[0]} et al.`
          : d.data.authors.join(", ");
        const citations = d.data.citation_count != null ? `${d.data.citation_count} citations` : "";
        tooltipDiv.innerHTML = `<strong>${d.data.title}</strong><br/><span style="color:#6B7280;font-size:11px">${authors}${d.data.year ? ` · ${d.data.year}` : ""}${citations ? ` · ${citations}` : ""}</span>`;
        tooltipDiv.style.left = `${event.offsetX + 10}px`;
        tooltipDiv.style.top = `${event.offsetY - 10}px`;
        tooltipDiv.style.opacity = "1";
      })
      .on("mouseleave", () => {
        // Restore
        nodeMerged.transition().duration(200).attr("opacity", 1);
        linkMerged.transition().duration(200)
          .attr("stroke-opacity", 0.45)
          .style("stroke", "#9CA3AF");
        labelMerged.transition().duration(200).attr("opacity", 1);
        tooltipDiv.style.opacity = "0";
      });

    // ── Tick handler ──
    let lastPositionUpdate = 0;
    simulation.on("tick", () => {
      linkMerged
        .attr("x1", (d) => (d.source as SimNode).x!)
        .attr("y1", (d) => (d.source as SimNode).y!)
        .attr("x2", (d) => (d.target as SimNode).x!)
        .attr("y2", (d) => (d.target as SimNode).y!);

      nodeMerged.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);

      glowMerged.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);

      labelMerged.attr("x", (d) => d.x!).attr("y", (d) => d.y!);

      const now = Date.now();
      if (onPositionsUpdate && now - lastPositionUpdate > 500) {
        lastPositionUpdate = now;
        onPositionsUpdate(
          simulation.nodes().map((n) => ({ x: n.x!, y: n.y!, data: n.data }))
        );
      }
    });

    // ── Reheat gently ──
    simulation.alpha(0.3).restart();

    // ── Animate camera to focus node ──
    if (focusNodeId) {
      const focusSimNode = nodes.find((n) => n.id === focusNodeId);
      // Wait a bit for positions to stabilize, then pan
      const timer = d3.timer(() => {
        if (
          focusSimNode &&
          focusSimNode.x != null &&
          focusSimNode.y != null &&
          simulation.alpha() < 0.1
        ) {
          svg
            .transition()
            .duration(500)
            .call(
              zoom.transform,
              d3.zoomIdentity
                .translate(
                  width / 2 - focusSimNode.x,
                  height / 2 - focusSimNode.y,
                )
                .scale(1.2),
            );
          timer.stop();
        }
      }, 100);

      // Safety: stop timer after 5s no matter what
      setTimeout(() => timer.stop(), 5000);
    }
    // colorMode is deliberately NOT a dependency: it is read through colorModeRef and
    // repainted by the cheap effect below, so switching Year/Cluster/Quality never
    // re-enters the general-update pattern or reheats the force simulation.
  }, [data, focusNodeId]);

  // ──────────────────────────────────────────────────────────
  // Colour-mode effect: repaint fills only, never touch the simulation
  // ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!nodesGroupRef.current) return;
    // No-op unless colorMode actually changed: this effect also fires on data/focus
    // churn (same dep array as the mount-time data-sync effect), and it must never
    // repaint on those — see paintedColorModeRef's declaration above.
    if (paintedColorModeRef.current === colorMode) return;
    paintedColorModeRef.current = colorMode;

    const years = data.nodes.map((n) => n.year).filter((y): y is number => y != null);
    const yearDomain: [number, number] = years.length
      ? [Math.min(...years), Math.max(...years)]
      : [2000, 2025];

    nodesGroupRef.current
      .selectAll<SVGCircleElement, SimNode>("circle")
      // Named transition: it must never pre-empt the data-sync transition (line ~479,
      // unnamed) that sets r/stroke/stroke-width/stroke-dasharray on the same circles —
      // a same-named transition on the same element cancels the prior one instantly.
      .transition("colorRepaint")
      .duration(200)
      .style("fill", (d) =>
        nodeColor(d.data, d.id === focusNodeId, colorMode, yearDomain),
      );
  }, [colorMode, data, focusNodeId]);

  // ──────────────────────────────────────────────────────────
  // Search highlighting effect
  // ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!nodesGroupRef.current || !labelsGroupRef.current) return;
    if (!searchQuery) {
      nodesGroupRef.current.selectAll("circle").attr("opacity", 1);
      labelsGroupRef.current.selectAll("text").attr("opacity", 1);
      return;
    }
    const q = searchQuery.toLowerCase();
    nodesGroupRef.current.selectAll<SVGCircleElement, SimNode>("circle")
      .attr("opacity", (d) =>
        d.data.title.toLowerCase().includes(q) ||
        d.data.authors.some((a) => a.toLowerCase().includes(q))
          ? 1 : 0.15
      );
    labelsGroupRef.current.selectAll<SVGTextElement, SimNode>("text")
      .attr("opacity", (d) =>
        d.data.title.toLowerCase().includes(q) ||
        d.data.authors.some((a) => a.toLowerCase().includes(q))
          ? 1 : 0.1
      );
  }, [searchQuery]);

  return (
    <div ref={containerRef} className="relative w-full h-[600px]">
      <svg ref={svgRef} className="w-full h-full" />
      <GraphSidePanel
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
        onExpand={handleExpand}
        onAddToLibrary={onAddToLibrary}
        isExpanding={expandNode.isPending}
      />
    </div>
  );
}
