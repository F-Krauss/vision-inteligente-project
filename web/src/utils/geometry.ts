// Small geometry helpers shared by the polygon editors (segmenter mask editor
// and the new annotation canvas). Normalized coordinates are 0..1.

export type Point = { x: number; y: number };

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

// Serialize normalized points to an SVG `points` string in a 0..100 viewBox.
export function polygonPoints(points: Array<{ x: number; y: number }>) {
  return points.map((point) => `${point.x * 100},${point.y * 100}`).join(" ");
}

export function distanceToSegment(point: Point, start: Point, end: Point) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (dx === 0 && dy === 0) return Math.hypot(point.x - start.x, point.y - start.y);
  const t = clamp(((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy), 0, 1);
  return Math.hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy));
}

// Insert a new point into the polygon edge nearest to it (keeps the loop sane
// when the user clicks to add a vertex between two existing ones).
export function insertPointInClosestSegment(points: Point[], point: Point) {
  if (points.length < 2) return [...points, point];
  let insertAt = points.length;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < points.length; index += 1) {
    const start = points[index];
    const end = points[(index + 1) % points.length];
    const distance = distanceToSegment(point, start, end);
    if (distance < bestDistance) {
      bestDistance = distance;
      insertAt = index + 1;
    }
  }
  return [...points.slice(0, insertAt), point, ...points.slice(insertAt)];
}
