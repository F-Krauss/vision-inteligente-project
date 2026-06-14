// Photoshop-style annotation surface. The image fills the stage and lives in a
// zoom/pan "world"; the annotation overlay is drawn in screen coordinates so
// stroke thickness stays constant at any zoom — recognizable yet never hiding
// tiny parts. A live loupe magnifies whatever sits under the pointer for precise
// vertex placement. Tools: polygon (click vertices), rectangle (drag), select
// (move / edit vertices), pan.

import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Point } from "../utils/geometry";
import { clamp } from "../utils/geometry";
import {
  type AnnElement,
  type Category,
  type Importance,
  type Tool,
  bboxOfPolygon,
  importanceColor,
  newId,
  pointInPolygon,
  rectPolygon,
} from "./types";

type Props = {
  imageUrl: string;
  elements: AnnElement[];
  selectedId: string | null;
  highlightCategoryId: string | null;
  tool: Tool;
  importance: Importance;
  category: Category | null;
  evaluationZone?: Point[] | null;
  onChange: (elements: AnnElement[]) => void;
  onEvaluationZoneChange?: (zone: Point[]) => void;
  onSelect: (id: string | null) => void;
  onViewChange?: (zoomPct: number) => void;
  readOnly?: boolean;
};

type View = { scale: number; tx: number; ty: number };
type DragState =
  | { mode: "pan"; last: [number, number] }
  | { mode: "move"; id: string; startN: Point; orig: Point[] }
  | { mode: "vertex"; id: string; index: number }
  | { mode: "rect"; startN: Point }
  | null;

const VERTEX_TOL = 11; // screen px for grabbing a vertex handle

export default function AnnotateCanvas({
  imageUrl,
  elements,
  selectedId,
  highlightCategoryId,
  tool,
  importance,
  category,
  evaluationZone = null,
  onChange,
  onEvaluationZoneChange,
  onSelect,
  onViewChange,
  readOnly = false,
}: Props) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const sourceRef = useRef<HTMLCanvasElement | null>(null);
  const loupeRef = useRef<HTMLCanvasElement | null>(null);
  const pointers = useRef<Map<number, { x: number; y: number }>>(new Map());
  const pinch = useRef<{ dist: number; scale: number } | null>(null);
  const drag = useRef<DragState>(null);

  const [view, setView] = useState<View>({ scale: 1, tx: 0, ty: 0 });
  const [stageSize, setStageSize] = useState({ w: 1, h: 1 });
  const [aspect, setAspect] = useState(1.5);
  const [draftPoly, setDraftPoly] = useState<Point[]>([]);
  const [draftRect, setDraftRect] = useState<[number, number, number, number] | null>(null);
  const [loupe, setLoupe] = useState<{ n: Point; corner: "tl" | "tr" } | null>(null);
  const [dragMode, setDragMode] = useState<NonNullable<DragState>["mode"] | null>(null);

  const baseW = Math.max(1, Math.min(stageSize.w * 0.92, stageSize.h * 0.92 * aspect));
  const baseH = Math.max(1, baseW / aspect);

  function fittedView(scale = 1): View {
    return {
      scale,
      tx: (stageSize.w - baseW * scale) / 2,
      ty: (stageSize.h - baseH * scale) / 2,
    };
  }

  useEffect(() => {
    let cancelled = false;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (cancelled) return;
      setAspect(img.naturalWidth / Math.max(1, img.naturalHeight));
      const cap = 2400;
      const s = Math.min(1, cap / Math.max(img.naturalWidth, img.naturalHeight));
      const c = sourceRef.current || document.createElement("canvas");
      c.width = Math.round(img.naturalWidth * s);
      c.height = Math.round(img.naturalHeight * s);
      c.getContext("2d")!.drawImage(img, 0, 0, c.width, c.height);
      sourceRef.current = c;
    };
    img.src = imageUrl;
    return () => { cancelled = true; };
  }, [imageUrl]);

  useLayoutEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const measure = () => setStageSize({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => { onViewChange?.(Math.round(view.scale * 100)); }, [view.scale, onViewChange]);

  useEffect(() => {
    setView(fittedView(1));
  }, [imageUrl, aspect, stageSize.w, stageSize.h]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cancel an in-progress draft when the tool changes away from a draw tool.
  useEffect(() => {
    if (tool !== "polygon" && tool !== "zone" && draftPoly.length) setDraftPoly([]);
    if (tool !== "rect" && draftRect) setDraftRect(null);
  }, [tool]); // eslint-disable-line react-hooks/exhaustive-deps

  const screenOf = (n: Point): [number, number] => [
    view.tx + n.x * baseW * view.scale,
    view.ty + n.y * baseH * view.scale,
  ];
  const normOf = (sx: number, sy: number): Point => ({
    x: (sx - view.tx) / (baseW * view.scale),
    y: (sy - view.ty) / (baseH * view.scale),
  });

  function localXY(e: React.PointerEvent): [number, number] {
    const r = stageRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  function setDrag(next: DragState) {
    drag.current = next;
    setDragMode(next?.mode ?? null);
  }

  function zoomAround(sx: number, sy: number, factor: number) {
    setView((v) => {
      const scale = clamp(v.scale * factor, 0.5, 18);
      const k = scale / v.scale;
      return { scale, tx: sx - (sx - v.tx) * k, ty: sy - (sy - v.ty) * k };
    });
  }

  function onWheel(e: React.WheelEvent) {
    e.preventDefault();
    const [sx, sy] = localXY(e as unknown as React.PointerEvent);
    zoomAround(sx, sy, e.deltaY < 0 ? 1.15 : 1 / 1.15);
  }

  function hitElement(n: Point): AnnElement | null {
    for (let i = elements.length - 1; i >= 0; i -= 1) {
      if (pointInPolygon(n.x, n.y, elements[i].polygon)) return elements[i];
    }
    return null;
  }

  function hitVertex(n: Point, el: AnnElement): number | null {
    const [px, py] = screenOf(n);
    for (let i = 0; i < el.polygon.length; i += 1) {
      const [vx, vy] = screenOf(el.polygon[i]);
      if (Math.hypot(px - vx, py - vy) < VERTEX_TOL) return i;
    }
    return null;
  }

  function commitElement(polygon: Point[], shape: AnnElement["shape"]) {
    const el: AnnElement = {
      id: newId("el"),
      shape,
      polygon,
      bbox: bboxOfPolygon(polygon),
      categoryId: category?.id ?? null,
      categoryName: category?.name ?? null,
      importance,
    };
    onChange([...elements, el]);
    onSelect(el.id);
  }

  function finishPolygon() {
    if (draftPoly.length >= 3) {
      if (tool === "zone") onEvaluationZoneChange?.(draftPoly);
      else commitElement(draftPoly, "polygon");
    }
    setDraftPoly([]);
  }

  function onPointerDown(e: React.PointerEvent) {
    try { (e.target as Element).setPointerCapture?.(e.pointerId); } catch { /* synthetic / lost pointer */ }
    const [sx, sy] = localXY(e);
    pointers.current.set(e.pointerId, { x: sx, y: sy });
    if (pointers.current.size === 2) {
      const pts = [...pointers.current.values()];
      pinch.current = { dist: Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y), scale: view.scale };
      setDrag(null);
      setDraftRect(null);
      return;
    }
    const n = normOf(sx, sy);
    const corner: "tl" | "tr" = sx > stageSize.w / 2 ? "tl" : "tr";

    if (tool === "pan" || e.button === 1 || e.altKey) { setDrag({ mode: "pan", last: [sx, sy] }); return; }
    if (readOnly) { onSelect(hitElement(n)?.id ?? null); return; }

    if (tool === "polygon" || tool === "zone") {
      // Click near the first vertex closes the loop.
      if (draftPoly.length >= 3) {
        const [fx, fy] = screenOf(draftPoly[0]);
        if (Math.hypot(sx - fx, sy - fy) < VERTEX_TOL + 3) { finishPolygon(); return; }
      }
      setDraftPoly((cur) => [...cur, n]);
      setLoupe({ n, corner });
      return;
    }
    if (tool === "rect") {
      setDrag({ mode: "rect", startN: n });
      setDraftRect([n.x, n.y, n.x, n.y]);
      setLoupe({ n, corner });
      return;
    }
    // select
    const sel = selectedId ? elements.find((el) => el.id === selectedId) ?? null : null;
    if (sel) {
      const vi = hitVertex(n, sel);
      if (vi !== null) { setDrag({ mode: "vertex", id: sel.id, index: vi }); setLoupe({ n, corner }); return; }
    }
    const hit = hitElement(n);
    if (hit) {
      onSelect(hit.id);
      setDrag({ mode: "move", id: hit.id, startN: n, orig: hit.polygon.map((p) => ({ ...p })) });
      setLoupe({ n, corner });
    } else {
      onSelect(null);
      setDrag({ mode: "pan", last: [sx, sy] });
    }
  }

  function updateElement(id: string, polygon: Point[]) {
    onChange(elements.map((el) => (el.id === id ? { ...el, polygon, bbox: bboxOfPolygon(polygon) } : el)));
  }

  function onPointerMove(e: React.PointerEvent) {
    const [sx, sy] = localXY(e);
    if (pointers.current.has(e.pointerId)) pointers.current.set(e.pointerId, { x: sx, y: sy });

    if (pinch.current && pointers.current.size === 2) {
      const pts = [...pointers.current.values()];
      const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      const cx = (pts[0].x + pts[1].x) / 2, cy = (pts[0].y + pts[1].y) / 2;
      setView((v) => {
        const scale = clamp(pinch.current!.scale * (dist / (pinch.current!.dist || dist)), 0.5, 18);
        const k = scale / v.scale;
        return { scale, tx: cx - (cx - v.tx) * k, ty: cy - (cy - v.ty) * k };
      });
      return;
    }

    const d = drag.current;
    const n = normOf(sx, sy);
    const corner: "tl" | "tr" = sx > stageSize.w / 2 ? "tl" : "tr";

    if (tool === "polygon" && draftPoly.length) setLoupe({ n, corner });
    if (!d) return;

    if (d.mode === "pan") {
      const dx = sx - d.last[0], dy = sy - d.last[1];
      d.last = [sx, sy];
      setView((v) => ({ ...v, tx: v.tx + dx, ty: v.ty + dy }));
      return;
    }
    if (d.mode === "rect") {
      setDraftRect([d.startN.x, d.startN.y, n.x, n.y]);
      setLoupe({ n, corner });
      return;
    }
    if (d.mode === "vertex") {
      const el = elements.find((x) => x.id === d.id);
      if (!el) return;
      const pts = el.polygon.map((p, i) => (i === d.index ? n : p));
      updateElement(d.id, el.shape === "rect" ? resizeRect(el.polygon, d.index, n) : pts);
      setLoupe({ n, corner });
      return;
    }
    if (d.mode === "move") {
      const dx = n.x - d.startN.x, dy = n.y - d.startN.y;
      updateElement(d.id, d.orig.map((p) => ({ x: clamp(p.x + dx, 0, 1), y: clamp(p.y + dy, 0, 1) })));
      setLoupe({ n, corner });
    }
  }

  function onPointerUp(e: React.PointerEvent) {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size < 2) pinch.current = null;
    const d = drag.current;
    if (d?.mode === "rect" && draftRect) {
      const [x1, y1, x2, y2] = draftRect;
      if (Math.abs(x2 - x1) > 0.004 && Math.abs(y2 - y1) > 0.004) commitElement(rectPolygon(x1, y1, x2, y2), "rect");
      setDraftRect(null);
    }
    setDrag(null);
    if (pointers.current.size === 0 && tool !== "polygon") setLoupe(null);
  }

  function onDoubleClick() { if (tool === "polygon") finishPolygon(); }

  // Loupe rendering (mirrors the lab canvas).
  useEffect(() => {
    const c = loupeRef.current;
    const src = sourceRef.current;
    if (!c || !src || !loupe) return;
    const ctx = c.getContext("2d")!;
    const R = c.width;
    const zoom = 6;
    const cxp = loupe.n.x * src.width;
    const cyp = loupe.n.y * src.height;
    const half = R / (2 * zoom);
    ctx.clearRect(0, 0, R, R);
    ctx.save();
    ctx.beginPath();
    ctx.arc(R / 2, R / 2, R / 2, 0, Math.PI * 2);
    ctx.clip();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(src, cxp - half, cyp - half, half * 2, half * 2, 0, 0, R, R);
    ctx.strokeStyle = "rgba(255,80,80,0.9)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(R / 2, R / 2 - 12); ctx.lineTo(R / 2, R / 2 + 12);
    ctx.moveTo(R / 2 - 12, R / 2); ctx.lineTo(R / 2 + 12, R / 2);
    ctx.stroke();
    ctx.restore();
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(R / 2, R / 2, R / 2 - 1, 0, Math.PI * 2);
    ctx.stroke();
  }, [loupe]);

  const selected = selectedId ? elements.find((el) => el.id === selectedId) ?? null : null;
  const draftScreen = useMemo(() => draftPoly.map((p) => screenOf(p)), [draftPoly, view, baseW, baseH]);
  const cursor = cursorFor(tool, dragMode);

  return (
    <div className="annStage" ref={stageRef}
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDoubleClick={onDoubleClick}
      style={{ cursor, touchAction: "none" }}
    >
      <div className="annWorld" style={{ transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`, width: baseW, height: baseH }}>
        <img src={imageUrl} alt="anotación" draggable={false} style={{ width: baseW, height: baseH }} />
      </div>

      <svg className="annOverlay" width={stageSize.w} height={stageSize.h}>
        {evaluationZone?.length ? (
          <polygon
            points={evaluationZone.map((p) => screenOf(p)).map((p) => `${p[0]},${p[1]}`).join(" ")}
            fill="none"
            stroke="#2f80ed"
            strokeWidth={3}
            strokeDasharray="10 5"
          />
        ) : null}

        {elements.map((el) => {
          const pts = el.polygon.map((p) => screenOf(p));
          const color = importanceColor(el.importance);
          const highlit = highlightCategoryId != null && el.categoryId === highlightCategoryId;
          const isSel = el.id === selectedId;
          return (
            <g key={el.id}>
              <polygon
                points={pts.map((p) => `${p[0]},${p[1]}`).join(" ")}
                fill={color}
                fillOpacity={highlit ? 0.28 : isSel ? 0.2 : 0.12}
                stroke={color}
                strokeWidth={isSel ? 3 : highlit ? 3 : 2}
                strokeDasharray={isSel ? "6 3" : undefined}
              />
              {isSel ? pts.map((p, i) => (
                <rect key={i} className="annHandle" x={p[0] - 5} y={p[1] - 5} width={10} height={10} />
              )) : null}
            </g>
          );
        })}

        {draftScreen.length ? (
          <g>
            <polyline points={draftScreen.map((p) => `${p[0]},${p[1]}`).join(" ")}
              fill="none" stroke={tool === "zone" ? "#2f80ed" : "#2563eb"} strokeWidth={2.5} strokeDasharray="5 4" />
            {draftScreen.map((p, i) => (
              <rect key={i} className={i === 0 ? "annHandle first" : "annHandle"} x={p[0] - 5} y={p[1] - 5} width={10} height={10} />
            ))}
          </g>
        ) : null}

        {draftRect ? (() => {
          const a = screenOf({ x: draftRect[0], y: draftRect[1] });
          const b = screenOf({ x: draftRect[2], y: draftRect[3] });
          return <rect x={Math.min(a[0], b[0])} y={Math.min(a[1], b[1])} width={Math.abs(b[0] - a[0])} height={Math.abs(b[1] - a[1])}
            fill="rgba(37,99,235,0.12)" stroke="#2563eb" strokeWidth={2} strokeDasharray="5 4" />;
        })() : null}
      </svg>

      {(tool === "polygon" || tool === "zone") && draftPoly.length ? (
        <div className="annDraftControls">
          <button type="button" onClick={finishPolygon} disabled={draftPoly.length < 3}>{tool === "zone" ? "Cerrar zona" : "Cerrar"} ({draftPoly.length})</button>
          <button type="button" onClick={() => setDraftPoly((c) => c.slice(0, -1))}>↶ Punto</button>
          <button type="button" onClick={() => setDraftPoly([])}>✕</button>
        </div>
      ) : null}

      {loupe ? (
        <div className={`annLoupe ${loupe.corner}`}><canvas ref={loupeRef} width={150} height={150} /></div>
      ) : null}

      <div className="annZoomCtl">
        <button type="button" onClick={() => zoomAround(stageSize.w / 2, stageSize.h / 2, 1.3)} title="Acercar">＋</button>
        <button type="button" onClick={() => zoomAround(stageSize.w / 2, stageSize.h / 2, 1 / 1.3)} title="Alejar">－</button>
        <button type="button" onClick={() => setView(fittedView(1))} title="Ajustar">⤢</button>
        <span className="annZoomBadge">{Math.round(view.scale * 100)}%</span>
      </div>
      {selected ? null : null}
    </div>
  );
}

function cursorFor(tool: Tool, dragMode: NonNullable<DragState>["mode"] | null): React.CSSProperties["cursor"] {
  if (dragMode === "pan") return "grabbing";
  if (dragMode === "move") return "move";
  if (dragMode === "vertex") return "crosshair";
  if (dragMode === "rect") return "crosshair";
  if (tool === "pan") return "grab";
  if (tool === "select") return "default";
  return "crosshair";
}

// Keep a rectangle rectangular while dragging corner `index` (order tl,tr,br,bl):
// move the dragged corner and its two neighbours so the opposite corner stays put.
function resizeRect(poly: Point[], index: number, n: Point): Point[] {
  const opp = poly[(index + 2) % 4];
  return rectPolygon(opp.x, opp.y, n.x, n.y);
}
