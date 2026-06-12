// Comfortable, touch-friendly annotation surface built for tiny parts:
// pinch / wheel zoom, drag pan, a live magnifier loupe that shows exactly what
// sits under the finger, plus zone-polygon and part-box tools with edit/resize.

import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Box, Poly } from "./vision";
import type { Part, PartSize } from "./store";
import { newId } from "./store";

type Tool = "part" | "zone" | "select" | "pan";

type Props = {
  imageUrl: string;
  zone: Poly | null;
  parts: Part[];
  partVerdicts?: Record<string, "present" | "missing" | "uncertain">;
  onZoneChange: (zone: Poly | null) => void;
  onPartsChange: (parts: Part[]) => void;
  readOnly?: boolean;
};

type View = { scale: number; tx: number; ty: number };

const SIZE_DEFAULT: PartSize = "small";

export default function AnnotationCanvas({
  imageUrl,
  zone,
  parts,
  partVerdicts,
  onZoneChange,
  onPartsChange,
  readOnly = false,
}: Props) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const sourceRef = useRef<HTMLCanvasElement | null>(null);
  const loupeRef = useRef<HTMLCanvasElement | null>(null);
  const pointers = useRef<Map<number, { x: number; y: number }>>(new Map());
  const pinch = useRef<{ dist: number; cx: number; cy: number; scale: number } | null>(null);
  const drag = useRef<{ mode: string; startN: [number, number]; orig?: Box; partId?: string; handle?: number } | null>(null);

  const [tool, setTool] = useState<Tool>("part");
  const [view, setView] = useState<View>({ scale: 1, tx: 0, ty: 0 });
  const [stageSize, setStageSize] = useState({ w: 1, h: 1 });
  const [aspect, setAspect] = useState(1.5);
  const [draft, setDraft] = useState<Box | null>(null);
  const [zoneDraft, setZoneDraft] = useState<Poly>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loupe, setLoupe] = useState<{ n: [number, number]; corner: "tl" | "tr" } | null>(null);
  const [label, setLabel] = useState("pieza");
  const [size, setSize] = useState<PartSize>(SIZE_DEFAULT);

  // Base displayed image size (scale 1) fills the stage width.
  const baseW = stageSize.w;
  const baseH = stageSize.w / aspect;

  // Load source image into an offscreen canvas for the loupe and measure aspect.
  useEffect(() => {
    let cancelled = false;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (cancelled) return;
      setAspect(img.naturalWidth / Math.max(1, img.naturalHeight));
      const cap = 2200;
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

  const screenOf = (n: [number, number]): [number, number] => [
    view.tx + n[0] * baseW * view.scale,
    view.ty + n[1] * baseH * view.scale,
  ];
  const normOf = (sx: number, sy: number): [number, number] => [
    (sx - view.tx) / (baseW * view.scale),
    (sy - view.ty) / (baseH * view.scale),
  ];

  function localXY(e: React.PointerEvent): [number, number] {
    const r = stageRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  function zoomAround(sx: number, sy: number, factor: number) {
    setView((v) => {
      const scale = clamp(v.scale * factor, 0.5, 12);
      const k = scale / v.scale;
      return { scale, tx: sx - (sx - v.tx) * k, ty: sy - (sy - v.ty) * k };
    });
  }

  function onWheel(e: React.WheelEvent) {
    e.preventDefault();
    const [sx, sy] = [e.clientX - stageRef.current!.getBoundingClientRect().left, e.clientY - stageRef.current!.getBoundingClientRect().top];
    zoomAround(sx, sy, e.deltaY < 0 ? 1.15 : 1 / 1.15);
  }

  function hitPart(n: [number, number]): Part | null {
    for (let i = parts.length - 1; i >= 0; i -= 1) {
      const b = parts[i].bbox;
      if (n[0] >= b[0] && n[0] <= b[2] && n[1] >= b[1] && n[1] <= b[3]) return parts[i];
    }
    return null;
  }

  function hitHandle(n: [number, number], part: Part): number | null {
    const corners: Array<[number, number]> = [
      [part.bbox[0], part.bbox[1]],
      [part.bbox[2], part.bbox[1]],
      [part.bbox[2], part.bbox[3]],
      [part.bbox[0], part.bbox[3]],
    ];
    const tol = 16 / (baseW * view.scale);
    for (let i = 0; i < 4; i += 1) {
      if (Math.abs(n[0] - corners[i][0]) < tol && Math.abs(n[1] - corners[i][1]) < tol) return i;
    }
    return null;
  }

  function onPointerDown(e: React.PointerEvent) {
    if (readOnly && tool !== "pan") return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const [sx, sy] = localXY(e);
    pointers.current.set(e.pointerId, { x: sx, y: sy });
    if (pointers.current.size === 2) {
      const pts = [...pointers.current.values()];
      const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      pinch.current = { dist, cx: (pts[0].x + pts[1].x) / 2, cy: (pts[0].y + pts[1].y) / 2, scale: view.scale };
      drag.current = null;
      setDraft(null);
      return;
    }
    const n = normOf(sx, sy);
    const corner: "tl" | "tr" = sx > stageSize.w / 2 ? "tl" : "tr";
    if (tool === "pan") {
      drag.current = { mode: "pan", startN: [sx, sy] };
      return;
    }
    if (tool === "part") {
      drag.current = { mode: "draw", startN: n };
      setDraft([n[0], n[1], n[0], n[1]]);
      setLoupe({ n, corner });
      return;
    }
    if (tool === "zone") {
      setZoneDraft((cur) => [...cur, n]);
      setLoupe({ n, corner });
      return;
    }
    // select
    const sel = selectedId ? parts.find((p) => p.id === selectedId) || null : null;
    if (sel) {
      const handle = hitHandle(n, sel);
      if (handle !== null) {
        drag.current = { mode: "resize", startN: n, orig: [...sel.bbox] as Box, partId: sel.id, handle };
        setLoupe({ n, corner });
        return;
      }
    }
    const hit = hitPart(n);
    if (hit) {
      setSelectedId(hit.id);
      drag.current = { mode: "move", startN: n, orig: [...hit.bbox] as Box, partId: hit.id };
      setLoupe({ n, corner });
    } else {
      setSelectedId(null);
    }
  }

  function onPointerMove(e: React.PointerEvent) {
    const [sx, sy] = localXY(e);
    if (pointers.current.has(e.pointerId)) pointers.current.set(e.pointerId, { x: sx, y: sy });

    if (pinch.current && pointers.current.size === 2) {
      const pts = [...pointers.current.values()];
      const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      const factor = dist / (pinch.current.dist || dist);
      const cx = (pts[0].x + pts[1].x) / 2;
      const cy = (pts[0].y + pts[1].y) / 2;
      setView((v) => {
        const scale = clamp(pinch.current!.scale * factor, 0.5, 12);
        const k = scale / v.scale;
        return { scale, tx: cx - (cx - v.tx) * k, ty: cy - (cy - v.ty) * k };
      });
      return;
    }

    const d = drag.current;
    if (!d) return;
    const n = normOf(sx, sy);
    const corner: "tl" | "tr" = sx > stageSize.w / 2 ? "tl" : "tr";
    if (d.mode === "pan") {
      const dx = sx - d.startN[0];
      const dy = sy - d.startN[1];
      d.startN = [sx, sy];
      setView((v) => ({ ...v, tx: v.tx + dx, ty: v.ty + dy }));
      return;
    }
    if (d.mode === "draw") {
      setDraft([d.startN[0], d.startN[1], n[0], n[1]]);
      setLoupe({ n, corner });
      return;
    }
    if (d.mode === "move" && d.orig && d.partId) {
      const dx = n[0] - d.startN[0];
      const dy = n[1] - d.startN[1];
      const b: Box = [d.orig[0] + dx, d.orig[1] + dy, d.orig[2] + dx, d.orig[3] + dy];
      onPartsChange(parts.map((p) => (p.id === d.partId ? { ...p, bbox: clampBox(b) } : p)));
      setLoupe({ n, corner });
      return;
    }
    if (d.mode === "resize" && d.orig && d.partId && d.handle !== undefined) {
      const b: Box = [...d.orig] as Box;
      if (d.handle === 0) { b[0] = n[0]; b[1] = n[1]; }
      else if (d.handle === 1) { b[2] = n[0]; b[1] = n[1]; }
      else if (d.handle === 2) { b[2] = n[0]; b[3] = n[1]; }
      else { b[0] = n[0]; b[3] = n[1]; }
      onPartsChange(parts.map((p) => (p.id === d.partId ? { ...p, bbox: normBox(b) } : p)));
      setLoupe({ n, corner });
    }
  }

  function onPointerUp(e: React.PointerEvent) {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size < 2) pinch.current = null;
    const d = drag.current;
    if (d?.mode === "draw" && draft) {
      const b = normBox(draft);
      if (b[2] - b[0] > 0.004 && b[3] - b[1] > 0.004) {
        onPartsChange([...parts, { id: newId("part"), label, bbox: b, size }]);
      }
    }
    drag.current = null;
    setDraft(null);
    if (pointers.current.size === 0) setLoupe(null);
  }

  // Draw the loupe whenever its target moves.
  useEffect(() => {
    const c = loupeRef.current;
    const src = sourceRef.current;
    if (!c || !src || !loupe) return;
    const ctx = c.getContext("2d")!;
    const R = c.width;
    const zoom = 6;
    const sw = src.width / (aspect >= 1 ? 1 : 1);
    const cxp = loupe.n[0] * src.width;
    const cyp = loupe.n[1] * src.height;
    const half = R / (2 * zoom);
    ctx.clearRect(0, 0, R, R);
    ctx.save();
    ctx.beginPath();
    ctx.arc(R / 2, R / 2, R / 2, 0, Math.PI * 2);
    ctx.clip();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(src, cxp - half, cyp - half, half * 2, half * 2, 0, 0, R, R);
    // crosshair
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
    void sw;
  }, [loupe, aspect]);

  function fit() {
    setView({ scale: 1, tx: 0, ty: 0 });
  }

  function closeZone() {
    if (zoneDraft.length >= 3) onZoneChange(zoneDraft);
    setZoneDraft([]);
  }

  function clearZone() {
    setZoneDraft([]);
    onZoneChange(null);
  }

  const zonePts = useMemo(() => {
    const poly = zoneDraft.length ? zoneDraft : zone || [];
    return poly.map((p) => screenOf(p));
  }, [zoneDraft, zone, view, baseW, baseH]);

  const selected = selectedId ? parts.find((p) => p.id === selectedId) || null : null;

  return (
    <div className="labAnnotator">
      {!readOnly ? (
        <div className="labTools">
          <div className="labToolGroup">
            <button className={tool === "part" ? "on" : ""} onClick={() => setTool("part")} type="button">▭ Pieza</button>
            <button className={tool === "zone" ? "on" : ""} onClick={() => setTool("zone")} type="button">⬡ Zona</button>
            <button className={tool === "select" ? "on" : ""} onClick={() => setTool("select")} type="button">✶ Editar</button>
            <button className={tool === "pan" ? "on" : ""} onClick={() => setTool("pan")} type="button">✋ Mover</button>
          </div>
          <div className="labToolGroup">
            <button onClick={() => zoomAround(stageSize.w / 2, stageSize.h / 2, 1.3)} type="button">＋</button>
            <button onClick={() => zoomAround(stageSize.w / 2, stageSize.h / 2, 1 / 1.3)} type="button">－</button>
            <button onClick={fit} type="button">Ajustar</button>
          </div>
          {tool === "part" ? (
            <div className="labToolGroup grow">
              <input className="labLabelInput" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Etiqueta" />
              <select value={size} onChange={(e) => setSize(e.target.value as PartSize)}>
                <option value="tiny">diminuta</option>
                <option value="small">pequeña</option>
                <option value="normal">normal</option>
              </select>
            </div>
          ) : null}
          {tool === "zone" ? (
            <div className="labToolGroup grow">
              <span className="labHint">Toca para añadir vértices ({zoneDraft.length})</span>
              <button onClick={closeZone} type="button" disabled={zoneDraft.length < 3}>Cerrar zona</button>
              <button onClick={() => setZoneDraft((c) => c.slice(0, -1))} type="button" disabled={!zoneDraft.length}>Deshacer</button>
              <button onClick={clearZone} type="button">Borrar zona</button>
            </div>
          ) : null}
          {tool === "select" && selected ? (
            <div className="labToolGroup grow">
              <input className="labLabelInput" value={selected.label} onChange={(e) => onPartsChange(parts.map((p) => p.id === selected.id ? { ...p, label: e.target.value } : p))} />
              <button className="danger" onClick={() => { onPartsChange(parts.filter((p) => p.id !== selected.id)); setSelectedId(null); }} type="button">Borrar pieza</button>
            </div>
          ) : null}
        </div>
      ) : null}

      <div
        ref={stageRef}
        className="labStage"
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        style={{ touchAction: "none" }}
      >
        <div
          className="labWorld"
          style={{
            transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`,
            width: baseW,
            height: baseH,
          }}
        >
          <img src={imageUrl} alt="referencia" draggable={false} style={{ width: baseW, height: baseH }} />
        </div>

        <svg className="labOverlay" width={stageSize.w} height={stageSize.h}>
          {zonePts.length >= 2 ? (
            <polygon
              className="labZone"
              points={zonePts.map((p) => `${p[0]},${p[1]}`).join(" ")}
            />
          ) : null}
          {zoneDraft.map((p, i) => {
            const s = screenOf(p);
            return <circle key={i} cx={s[0]} cy={s[1]} r={5} className="labVertex" />;
          })}
          {parts.map((part) => {
            const a = screenOf([part.bbox[0], part.bbox[1]]);
            const b = screenOf([part.bbox[2], part.bbox[3]]);
            const verdict = partVerdicts?.[part.id];
            const cls = verdict ? `labPart v-${verdict}` : `labPart ${selectedId === part.id ? "sel" : ""}`;
            return (
              <g key={part.id}>
                <rect className={cls} x={a[0]} y={a[1]} width={b[0] - a[0]} height={b[1] - a[1]} />
                <text className="labPartLabel" x={a[0] + 2} y={a[1] - 3}>{part.label}</text>
              </g>
            );
          })}
          {selected ? [0, 1, 2, 3].map((i) => {
            const corners: Array<[number, number]> = [
              [selected.bbox[0], selected.bbox[1]],
              [selected.bbox[2], selected.bbox[1]],
              [selected.bbox[2], selected.bbox[3]],
              [selected.bbox[0], selected.bbox[3]],
            ];
            const s = screenOf(corners[i]);
            return <rect key={i} className="labHandle" x={s[0] - 7} y={s[1] - 7} width={14} height={14} />;
          }) : null}
          {draft ? (() => {
            const a = screenOf([draft[0], draft[1]]);
            const b = screenOf([draft[2], draft[3]]);
            return <rect className="labPart draft" x={Math.min(a[0], b[0])} y={Math.min(a[1], b[1])} width={Math.abs(b[0] - a[0])} height={Math.abs(b[1] - a[1])} />;
          })() : null}
        </svg>

        {loupe ? (
          <div className={`labLoupe ${loupe.corner}`}>
            <canvas ref={loupeRef} width={150} height={150} />
          </div>
        ) : null}

        <div className="labZoomBadge">{Math.round(view.scale * 100)}%</div>
      </div>
    </div>
  );
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}
function normBox(b: Box): Box {
  return [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])].map((v) => clamp(v, 0, 1)) as Box;
}
function clampBox(b: Box): Box {
  return [clamp(b[0], 0, 1), clamp(b[1], 0, 1), clamp(b[2], 0, 1), clamp(b[3], 0, 1)] as Box;
}
