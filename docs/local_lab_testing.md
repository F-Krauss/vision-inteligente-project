# Prueba local — manual testing guide

A fully client-side bench (no backend, no cloud, no models) to try the whole
concept on simple objects at home: register a reference, mark the parts, capture
with a guided ghost overlay, and get a per-part present/missing verdict. Lives in
`web/src/lab/`, reachable from the sidebar entry **"Prueba local"** (it is the
default view).

## Run it

```bash
cd web
npm install      # first time only
npm run dev      # opens on http://localhost:5173
```

Open `http://localhost:5173` — "Prueba local" loads first.

### Camera works on `localhost` only (security rule)

Browsers block the camera on plain `http://` LAN addresses. So:

- On the **laptop** (webcam): `http://localhost:5173` works fully, camera included.
  This is the easiest way to validate the flow with simple objects today.
- On a **tablet**, the camera needs HTTPS. Two options:
  - Build and deploy the static site (`npm run build` → host `web/dist` on any
    HTTPS URL). The lab is 100% client-side, so no API is required.
  - Or serve the dev server over HTTPS / through an HTTPS tunnel.

Uploading a photo from the gallery works everywhere (no camera permission), so
you can also test the comparison without live capture.

## The test loop

1. **Nueva referencia** → take/upload a photo of the correct setup (all parts
   present). Lay your objects out (screws, coins, LEGO, whatever) and shoot once.
2. **Anotar**:
   - `⬡ Zona`: tap vertices around the area to check, then "Cerrar zona".
   - `▭ Pieza`: drag a box around each object. Pinch / wheel to zoom; the
     **loupe** (top corner) shows what's under your finger so tiny parts are easy.
   - `✶ Editar`: select a box to move/resize/rename/delete.
   - "Guardar" (persists to the device via localStorage).
3. **Inspeccionar** → guided capture opens. Match the live view to the green
   ghost of your reference. The ring shows alignment %; when it stays above the
   threshold it auto-captures. "Exigencia" sets how strict the match must be;
   "Transparencia" fades the ghost. You can also "Capturar ahora".
4. **Results**: each part is colored green/amber/red on the reference, with a
   list of present / revisar / falta and a per-part change %. The **Sensibilidad**
   slider re-scores instantly (no re-capture) — slide it to find the point that
   separates your present vs missing objects.

## What "good" looks like (validated on the mold data)

- The same photo compared to itself → everything **present**, ~0% change.
- A removed part at a matched pose → **falta**, with a clear change jump
  (~30%+) vs intact parts (~0–10%).
- If the framing is too different to locate a part, it returns **revisar**
  (uncertain) instead of guessing — by design (prefer a human check over a
  wrong pass).

## What to report back

For each try: object type, how many present/missing it called correctly, any
false "falta" on present parts, and the Sensibilidad value that worked best.
That tells us the threshold to bake in and whether the guided overlay holds pose
well enough on your tablet.

## Notes / limits (same as the factory findings)

- Tiny parts need to be a meaningful fraction of the frame — get closer or split
  the scene into sections rather than shooting everything from far away.
- Guided capture is what makes this work: it forces the inspection pose to match
  the reference, which is the single thing that makes small-part detection
  reliable. Free-pose, far-away photos will produce "revisar".
