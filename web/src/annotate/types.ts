// Data model for the annotation workspace. Coordinates are normalized 0..1 so
// they survive resize/zoom and transfer across differently-sized photos.

import type { Point } from "../utils/geometry";

export type Importance = "critical" | "relevant" | "minor"; // red / yellow / green

export const IMPORTANCES: Array<{ value: Importance; label: string; color: string }> = [
  { value: "critical", label: "Crítica", color: "#dc2626" },
  { value: "relevant", label: "Relevante", color: "#f59e0b" },
  { value: "minor", label: "No relevante", color: "#22c55e" },
];

export function importanceColor(importance: Importance): string {
  return IMPORTANCES.find((entry) => entry.value === importance)?.color ?? "#f59e0b";
}

export function importanceLabel(importance: Importance): string {
  return IMPORTANCES.find((entry) => entry.value === importance)?.label ?? "Relevante";
}

export type Category = {
  id: string;
  name: string;
  slug: string;
  color?: string;
};

export type Shape = "polygon" | "rect";

// One mapped part. `polygon` holds the vertices (4 corners for a rectangle);
// `bbox` is the derived axis-aligned bounds, kept in sync for hit-tests and so
// the backend/training pipeline always has a box even for polygon shapes.
export type AnnElement = {
  id: string;
  shape: Shape;
  polygon: Point[];
  bbox: [number, number, number, number];
  categoryId: string | null;
  categoryName: string | null;
  importance: Importance;
  notes?: string;
};

export type Tool = "zone" | "polygon" | "rect" | "select" | "pan";

export function newId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
}

export function slugify(name: string): string {
  return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "categoria";
}

export function bboxOfPolygon(points: Point[]): [number, number, number, number] {
  if (!points.length) return [0, 0, 0, 0];
  let x1 = 1, y1 = 1, x2 = 0, y2 = 0;
  for (const p of points) {
    x1 = Math.min(x1, p.x); y1 = Math.min(y1, p.y);
    x2 = Math.max(x2, p.x); y2 = Math.max(y2, p.y);
  }
  return [x1, y1, x2, y2];
}

export function rectPolygon(x1: number, y1: number, x2: number, y2: number): Point[] {
  const ax = Math.min(x1, x2), ay = Math.min(y1, y2), bx = Math.max(x1, x2), by = Math.max(y1, y2);
  return [{ x: ax, y: ay }, { x: bx, y: ay }, { x: bx, y: by }, { x: ax, y: by }];
}

export function polygonArea(points: Point[]): number {
  let area = 0;
  for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
    area += (points[j].x + points[i].x) * (points[j].y - points[i].y);
  }
  return Math.abs(area) / 2;
}

export function pointInPolygon(x: number, y: number, points: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
    const xi = points[i].x, yi = points[i].y, xj = points[j].x, yj = points[j].y;
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}
