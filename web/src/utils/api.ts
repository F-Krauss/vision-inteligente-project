// Shared API plumbing for the app: talks to the FastAPI backend (Cloud Run in
// prod, uvicorn locally) and resolves object-storage URLs. Extracted from
// main.tsx so the annotation module and the rest of the app share one source.

export const API_BASE = import.meta.env.VITE_API_BASE || "";

export function isLocalRuntime() {
  const apiBase = API_BASE.toLowerCase();
  const hostname = typeof window === "undefined" ? "" : window.location.hostname.toLowerCase();
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "::1" ||
    apiBase.includes("localhost") ||
    apiBase.includes("127.0.0.1") ||
    apiBase.includes("[::1]")
  );
}

export function resolveUrl(path: string) {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

export function displayUrl(path: string) {
  if (!path) return "";
  if (path.startsWith("local://")) return resolveUrl(`/v1/uploads/${path.replace("local://", "")}/file`);
  if (path.startsWith("/")) return resolveUrl(path);
  return path;
}

export async function postJson(path: string, body: unknown) {
  const response = await fetch(resolveUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
  return response.json();
}

export async function getJson(path: string) {
  const response = await fetch(resolveUrl(path), { cache: "no-store" });
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
  return response.json();
}

// Presign → PUT to object storage → return the stable object URI.
export async function uploadSystemFile(file: File, family: string, zoneId: string, purpose: string) {
  const presign = await postJson("/v1/uploads/presign", {
    filename: file.name,
    content_type: file.type || "application/octet-stream",
    family,
    zone_id: zoneId,
    purpose,
  });
  await fetch(resolveUrl(presign.upload_url), { method: presign.method, headers: presign.headers, body: file });
  return presign.object_uri as string;
}

export async function uploadFiles(files: File[], family: string, zoneId: string, purpose: string) {
  return Promise.all(files.map((file) => uploadSystemFile(file, family, zoneId, purpose)));
}

export function messageFrom(error: unknown) {
  return error instanceof Error ? error.message : "Error inesperado.";
}
