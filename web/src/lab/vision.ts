// Client-side computer vision for the local lab. Everything runs in the browser
// with the Canvas 2D API — no backend, no models. The signals mirror the Python
// presence package: lighting-invariant interior-vs-ring darkness, edge-density
// change, and normalized appearance correlation, fused into a presence score.

export type Box = [number, number, number, number]; // x1,y1,x2,y2 normalized 0..1
export type Poly = Array<[number, number]>; // normalized vertices

export type GrayImage = {
  gray: Float32Array; // 0..255
  edge: Float32Array; // 0..~255 gradient magnitude
  width: number;
  height: number;
};

export type Transform = { dx: number; dy: number; scale: number }; // golden -> inspection, normalized dx/dy

export type AlignResult = Transform & { score: number };

export type PartSignals = {
  ringRatioGolden: number;
  ringRatioInsp: number;
  edgeGolden: number;
  edgeInsp: number;
  ncc: number;
  context: number; // how well the surrounding ring locked (0..1); low = unreliable
};

export type PartVerdict = {
  score: number; // presence 0..1 (1 = present)
  decision: "present" | "missing" | "uncertain";
  change: number; // 0..1 change evidence
  signals: PartSignals;
};

// ── Image loading ──────────────────────────────────────────────────────────────

export async function fileToDataUrl(file: File, maxSide = 1600, quality = 0.88): Promise<string> {
  const bitmap = await blobToBitmap(file);
  const { canvas } = drawScaled(bitmap, maxSide);
  bitmap.close?.();
  return canvas.toDataURL("image/jpeg", quality);
}

export async function loadGray(src: string | ImageBitmap, maxSide = 768): Promise<GrayImage> {
  const bitmap = typeof src === "string" ? await urlToBitmap(src) : src;
  const { ctx, width, height } = drawScaled(bitmap, maxSide);
  if (typeof src === "string") bitmap.close?.();
  const data = ctx.getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < gray.length; i += 1, p += 4) {
    gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }
  const edge = sobelMagnitude(gray, width, height);
  return { gray, edge, width, height };
}

export function imageDataToGray(data: Uint8ClampedArray, width: number, height: number): GrayImage {
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < gray.length; i += 1, p += 4) {
    gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }
  return { gray, edge: sobelMagnitude(gray, width, height), width, height };
}

async function blobToBitmap(blob: Blob): Promise<ImageBitmap> {
  if ("createImageBitmap" in window) return createImageBitmap(blob);
  const url = URL.createObjectURL(blob);
  try {
    return await urlToBitmap(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function urlToBitmap(url: string): Promise<ImageBitmap> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if ("createImageBitmap" in window) {
        createImageBitmap(img).then(resolve).catch(reject);
      } else {
        // Fallback: wrap in a canvas-backed bitmap-like via offscreen draw
        const c = document.createElement("canvas");
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
        c.getContext("2d")!.drawImage(img, 0, 0);
        resolve(img as unknown as ImageBitmap);
      }
    };
    img.onerror = () => reject(new Error("No se pudo cargar la imagen"));
    img.src = url;
  });
}

function drawScaled(bitmap: ImageBitmap | HTMLImageElement, maxSide: number) {
  const iw = (bitmap as ImageBitmap).width || (bitmap as HTMLImageElement).naturalWidth;
  const ih = (bitmap as ImageBitmap).height || (bitmap as HTMLImageElement).naturalHeight;
  const s = Math.min(1, maxSide / Math.max(iw, ih));
  const width = Math.max(1, Math.round(iw * s));
  const height = Math.max(1, Math.round(ih * s));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bitmap as CanvasImageSource, 0, 0, width, height);
  return { canvas, ctx, width, height };
}

// ── Gradient / edges ─────────────────────────────────────────────────────────

function sobelMagnitude(gray: Float32Array, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y += 1) {
    for (let x = 1; x < w - 1; x += 1) {
      const i = y * w + x;
      const tl = gray[i - w - 1], tc = gray[i - w], tr = gray[i - w + 1];
      const ml = gray[i - 1], mr = gray[i + 1];
      const bl = gray[i + w - 1], bc = gray[i + w], br = gray[i + w + 1];
      const gx = tl + 2 * ml + bl - tr - 2 * mr - br;
      const gy = tl + 2 * tc + tr - bl - 2 * bc - br;
      out[i] = Math.hypot(gx, gy) * 0.25;
    }
  }
  return out;
}

// ── Geometry ─────────────────────────────────────────────────────────────────

export function pointInPoly(x: number, y: number, poly: Poly): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

export function polyBounds(poly: Poly): Box {
  let x1 = 1, y1 = 1, x2 = 0, y2 = 0;
  for (const [x, y] of poly) {
    x1 = Math.min(x1, x); y1 = Math.min(y1, y);
    x2 = Math.max(x2, x); y2 = Math.max(y2, y);
  }
  return [x1, y1, x2, y2];
}

function buildZoneMask(poly: Poly | null, w: number, h: number): Uint8Array {
  const mask = new Uint8Array(w * h);
  if (!poly || poly.length < 3) {
    mask.fill(1);
    return mask;
  }
  for (let y = 0; y < h; y += 1) {
    const ny = y / h;
    for (let x = 0; x < w; x += 1) {
      if (pointInPoly(x / w, ny, poly)) mask[y * w + x] = 1;
    }
  }
  return mask;
}

// ── Alignment (translation + scale, coarse-to-fine) ──────────────────────────

export function align(golden: GrayImage, insp: GrayImage, zone: Poly | null): AlignResult {
  const w = golden.width;
  const h = golden.height;
  const mask = buildZoneMask(zone, w, h);
  const cx = w / 2;
  const cy = h / 2;

  // Precompute golden edge energy within mask for normalization. Must use the
  // SAME 2px stride as the eval loop or the normalizer is ~4x too large.
  let gEnergy = 0;
  for (let y = 0; y < h; y += 2) {
    for (let x = 0; x < w; x += 2) {
      const i = y * w + x;
      if (mask[i]) gEnergy += golden.edge[i] * golden.edge[i];
    }
  }
  gEnergy = Math.sqrt(gEnergy) + 1e-6;

  const evalAt = (dx: number, dy: number, scale: number): number => {
    let dot = 0;
    let iEnergy = 0;
    const tx = dx * w;
    const ty = dy * h;
    // Sample every 2px for speed.
    for (let y = 0; y < h; y += 2) {
      for (let x = 0; x < w; x += 2) {
        const gi = y * w + x;
        if (!mask[gi]) continue;
        const sx = (x - cx) * scale + cx + tx;
        const sy = (y - cy) * scale + cy + ty;
        const ix = sx | 0;
        const iy = sy | 0;
        if (ix < 0 || iy < 0 || ix >= w || iy >= h) continue;
        const iv = insp.edge[iy * w + ix];
        const gv = golden.edge[gi];
        dot += gv * iv;
        iEnergy += iv * iv;
      }
    }
    return dot / (gEnergy * (Math.sqrt(iEnergy) + 1e-6));
  };

  let best: AlignResult = { dx: 0, dy: 0, scale: 1, score: evalAt(0, 0, 1) };
  // Coarse translation search at scale 1.
  for (let dy = -0.08; dy <= 0.08 + 1e-9; dy += 0.02) {
    for (let dx = -0.08; dx <= 0.08 + 1e-9; dx += 0.02) {
      const s = evalAt(dx, dy, 1);
      if (s > best.score) best = { dx, dy, scale: 1, score: s };
    }
  }
  // Refine translation + scale around best.
  const base = { ...best };
  for (const scale of [0.94, 0.97, 1.0, 1.03, 1.06]) {
    for (let dy = base.dy - 0.03; dy <= base.dy + 0.03 + 1e-9; dy += 0.01) {
      for (let dx = base.dx - 0.03; dx <= base.dx + 0.03 + 1e-9; dx += 0.01) {
        const s = evalAt(dx, dy, scale);
        if (s > best.score) best = { dx, dy, scale, score: s };
      }
    }
  }
  return best;
}

// Quick correlation-only score for live capture guidance (fixed coarse search).
export function quickAlignScore(golden: GrayImage, insp: GrayImage, zone: Poly | null): number {
  return align(golden, insp, zone).score;
}

export function mapBox(box: Box, t: Transform): Box {
  const cx = 0.5;
  const cy = 0.5;
  const map = (x: number, y: number): [number, number] => [
    (x - cx) * t.scale + cx + t.dx,
    (y - cy) * t.scale + cy + t.dy,
  ];
  const [ax, ay] = map(box[0], box[1]);
  const [bx, by] = map(box[2], box[3]);
  return [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)];
}

// ── Per-part signals ─────────────────────────────────────────────────────────

function medianInBox(img: Float32Array, w: number, h: number, box: Box): number {
  const x1 = Math.max(0, Math.floor(box[0] * w));
  const y1 = Math.max(0, Math.floor(box[1] * h));
  const x2 = Math.min(w, Math.ceil(box[2] * w));
  const y2 = Math.min(h, Math.ceil(box[3] * h));
  const vals: number[] = [];
  for (let y = y1; y < y2; y += 1) for (let x = x1; x < x2; x += 1) vals.push(img[y * w + x]);
  if (!vals.length) return 0;
  vals.sort((a, b) => a - b);
  return vals[vals.length >> 1];
}

function ringRatio(img: Float32Array, w: number, h: number, box: Box): number {
  const interior = medianInBox(img, w, h, box);
  const bw = box[2] - box[0];
  const bh = box[3] - box[1];
  const ring: Box = [
    box[0] - bw * 0.3,
    box[1] - bh * 0.3,
    box[2] + bw * 0.3,
    box[3] + bh * 0.3,
  ];
  // Ring median approximated by outer box median (dominated by surrounding metal).
  const outer = medianInBox(img, w, h, ring);
  return interior / Math.max(outer, 1);
}

function edgeDensity(edge: Float32Array, w: number, h: number, box: Box): number {
  const x1 = Math.max(0, Math.floor(box[0] * w));
  const y1 = Math.max(0, Math.floor(box[1] * h));
  const x2 = Math.min(w, Math.ceil(box[2] * w));
  const y2 = Math.min(h, Math.ceil(box[3] * h));
  let strong = 0;
  let total = 0;
  for (let y = y1; y < y2; y += 1) {
    for (let x = x1; x < x2; x += 1) {
      if (edge[y * w + x] >= 28) strong += 1;
      total += 1;
    }
  }
  return total ? strong / total : 0;
}

// Sample a box region into an NxN grid of gray values (bilinear-ish nearest).
function sampleGrid(img: Float32Array, w: number, h: number, box: Box, n: number): Float32Array {
  const out = new Float32Array(n * n);
  for (let gy = 0; gy < n; gy += 1) {
    const fy = box[1] + ((gy + 0.5) / n) * (box[3] - box[1]);
    const iy = Math.min(h - 1, Math.max(0, Math.round(fy * h)));
    for (let gx = 0; gx < n; gx += 1) {
      const fx = box[0] + ((gx + 0.5) / n) * (box[2] - box[0]);
      const ix = Math.min(w - 1, Math.max(0, Math.round(fx * w)));
      out[gy * n + gx] = img[iy * w + ix];
    }
  }
  return out;
}

function gridNcc(a: Float32Array, b: Float32Array): number {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i += 1) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i += 1) {
    const xa = a[i] - ma, xb = b[i] - mb;
    num += xa * xb; da += xa * xa; db += xb * xb;
  }
  if (da < 1e-6 || db < 1e-6) return 0;
  return num / Math.sqrt(da * db);
}

const CTX_N = 26;
const CTX_EXPAND = 1.8;
// Indices of grid cells that fall in the surrounding ring (outside the inner
// box) — used to lock position on stable context, not the (possibly changed)
// interior.
const CTX_RING_IDX = (() => {
  const inner = 1 / CTX_EXPAND; // inner box fraction of the expanded box
  const lo = 0.5 - inner / 2;
  const hi = 0.5 + inner / 2;
  const idx: number[] = [];
  for (let gy = 0; gy < CTX_N; gy += 1) {
    const py = (gy + 0.5) / CTX_N;
    for (let gx = 0; gx < CTX_N; gx += 1) {
      const px = (gx + 0.5) / CTX_N;
      const insideInner = px > lo && px < hi && py > lo && py < hi;
      if (!insideInner) idx.push(gy * CTX_N + gx);
    }
  }
  return idx;
})();

function expandBox(box: Box, factor: number): Box {
  const cx = (box[0] + box[2]) / 2;
  const cy = (box[1] + box[3]) / 2;
  const bw = (box[2] - box[0]) * factor;
  const bh = (box[3] - box[1]) * factor;
  return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2];
}

function ringValues(img: Float32Array, w: number, h: number, innerBox: Box): Float32Array {
  const grid = sampleGrid(img, w, h, expandBox(innerBox, CTX_EXPAND), CTX_N);
  const out = new Float32Array(CTX_RING_IDX.length);
  for (let i = 0; i < CTX_RING_IDX.length; i += 1) out[i] = grid[CTX_RING_IDX[i]];
  return out;
}

// Local refinement: slide the mapped box so the SURROUNDING RING best matches
// the golden's ring. Locking on stable context (not the interior) absorbs
// parallax while letting a removed interior still read as a cavity.
function refineBox(golden: GrayImage, insp: GrayImage, gBox: Box, iBox0: Box): { box: Box; ctx: number } {
  const gRing = ringValues(golden.gray, golden.width, golden.height, gBox);
  const range = 0.05;
  const step = 0.006;
  let best = { dx: 0, dy: 0, ctx: -2 };
  for (let dy = -range; dy <= range + 1e-9; dy += step) {
    for (let dx = -range; dx <= range + 1e-9; dx += step) {
      const shifted: Box = [iBox0[0] + dx, iBox0[1] + dy, iBox0[2] + dx, iBox0[3] + dy];
      const iRing = ringValues(insp.gray, insp.width, insp.height, shifted);
      const ctx = gridNcc(gRing, iRing);
      if (ctx > best.ctx) best = { dx, dy, ctx };
    }
  }
  return {
    box: [iBox0[0] + best.dx, iBox0[1] + best.dy, iBox0[2] + best.dx, iBox0[3] + best.dy],
    ctx: best.ctx,
  };
}

function interiorNcc(golden: GrayImage, insp: GrayImage, gBox: Box, iBox: Box): number {
  const gw = golden.width, gh = golden.height;
  const sx1 = Math.max(0, Math.floor(gBox[0] * gw));
  const sy1 = Math.max(0, Math.floor(gBox[1] * gh));
  const sx2 = Math.min(gw, Math.ceil(gBox[2] * gw));
  const sy2 = Math.min(gh, Math.ceil(gBox[3] * gh));
  const bw = sx2 - sx1;
  const bh = sy2 - sy1;
  if (bw < 3 || bh < 3) return 1;
  const iw = insp.width, ih = insp.height;
  const a: number[] = [];
  const b: number[] = [];
  for (let y = 0; y < bh; y += 1) {
    for (let x = 0; x < bw; x += 1) {
      a.push(golden.gray[(sy1 + y) * gw + (sx1 + x)]);
      // Sample inspection at the mapped sub-pixel position.
      const fx = iBox[0] + ((x + 0.5) / bw) * (iBox[2] - iBox[0]);
      const fy = iBox[1] + ((y + 0.5) / bh) * (iBox[3] - iBox[1]);
      const ix = Math.min(iw - 1, Math.max(0, Math.round(fx * iw)));
      const iy = Math.min(ih - 1, Math.max(0, Math.round(fy * ih)));
      b.push(insp.gray[iy * iw + ix]);
    }
  }
  return pearson(a, b);
}

function pearson(a: number[], b: number[]): number {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i += 1) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i += 1) {
    const xa = a[i] - ma;
    const xb = b[i] - mb;
    num += xa * xb; da += xa * xa; db += xb * xb;
  }
  if (da < 1e-6 || db < 1e-6) return 0;
  return num / Math.sqrt(da * db);
}

export function partSignals(golden: GrayImage, insp: GrayImage, gBox: Box, t: Transform): PartSignals {
  const iBox0 = mapBox(gBox, t);
  // Lock position on the surrounding ring, then read interior signals there.
  const refined = refineBox(golden, insp, gBox, iBox0);
  const iBox = refined.box;
  return {
    ringRatioGolden: ringRatio(golden.gray, golden.width, golden.height, gBox),
    ringRatioInsp: ringRatio(insp.gray, insp.width, insp.height, iBox),
    edgeGolden: edgeDensity(golden.edge, golden.width, golden.height, gBox),
    edgeInsp: edgeDensity(insp.edge, insp.width, insp.height, iBox),
    // Interior appearance match at the context-locked position: an intact part
    // matches (high), a removed part leaves a mismatched cavity (low).
    ncc: interiorNcc(golden, insp, gBox, iBox),
    context: refined.ctx,
  };
}

// sensitivity 0..1 (higher = flags more aggressively)
export function scorePart(s: PartSignals, sensitivity = 0.5): PartVerdict {
  const eNcc = clamp01((0.62 - s.ncc) / 0.62); // ncc<0.62 starts to count as change
  const edgeDrop = (s.edgeGolden - s.edgeInsp) / Math.max(s.edgeGolden, 0.02);
  const eEdge = clamp01(edgeDrop / 0.6);
  const ringDelta = s.ringRatioGolden - s.ringRatioInsp; // positive = interior got darker
  const eRing = clamp01(ringDelta / 0.35);
  const change = clamp01(0.55 * eNcc + 0.3 * eEdge + 0.25 * eRing);
  const score = 1 - change;
  // Calibrated on real same-pose pairs: a removed part yields change ~0.3, an
  // intact part ~0.02-0.13. Higher sensitivity lowers the bar for "missing".
  const tMissing = 0.3 - sensitivity * 0.18; // 0.30 (lenient) .. 0.12 (strict)
  let decision: PartVerdict["decision"];
  if (s.context < 0.35) {
    // Couldn't reliably locate the part (pose too different / occluded): the
    // measurement isn't trustworthy, so defer to a human rather than guess.
    decision = "uncertain";
  } else if (change >= tMissing) {
    decision = "missing";
  } else if (change <= tMissing - 0.09) {
    decision = "present";
  } else {
    decision = "uncertain";
  }
  return { score, decision, change, signals: s };
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}
