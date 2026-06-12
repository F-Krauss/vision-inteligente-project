// LocalStorage-backed persistence for the lab. A "reference" bundles the golden
// image (as a JPEG data URL), the verifiable zone polygon, and the registered
// parts. Everything stays on-device so the lab works fully offline.

import type { Box, Poly } from "./vision";

export type PartSize = "tiny" | "small" | "normal";

export type Part = {
  id: string;
  label: string;
  bbox: Box;
  size: PartSize;
};

export type Reference = {
  id: string;
  name: string;
  createdAt: string;
  imageDataUrl: string;
  width: number;
  height: number;
  zone: Poly | null;
  parts: Part[];
};

const KEY = "moldvision.lab.references.v1";

export function loadReferences(): Reference[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Reference[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveReferences(refs: Reference[]): { ok: boolean; error?: string } {
  try {
    localStorage.setItem(KEY, JSON.stringify(refs));
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "No se pudo guardar (¿almacenamiento lleno?)" };
  }
}

export function upsertReference(ref: Reference): { ok: boolean; error?: string; refs: Reference[] } {
  const refs = loadReferences();
  const idx = refs.findIndex((r) => r.id === ref.id);
  if (idx >= 0) refs[idx] = ref;
  else refs.unshift(ref);
  const result = saveReferences(refs);
  return { ...result, refs };
}

export function deleteReference(id: string): Reference[] {
  const refs = loadReferences().filter((r) => r.id !== id);
  saveReferences(refs);
  return refs;
}

export function newId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
}
