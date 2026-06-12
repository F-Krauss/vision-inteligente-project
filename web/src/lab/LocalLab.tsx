// Self-contained local test bench: register a reference (golden) with a zone and
// parts, capture an inspection with guided alignment, and compare per-part — all
// in the browser, no backend. Built for trying the concept on simple objects.

import React, { useEffect, useMemo, useState } from "react";
import AnnotationCanvas from "./AnnotationCanvas";
import GuidedCapture from "./GuidedCapture";
import {
  deleteReference,
  loadReferences,
  newId,
  upsertReference,
  type Part,
  type Reference,
} from "./store";
import {
  align,
  fileToDataUrl,
  loadGray,
  partSignals,
  scorePart,
  type Poly,
  type PartVerdict,
} from "./vision";
import "./lab.css";

type Mode = "overview" | "annotate" | "results";

type Inspection = {
  inspUrl: string;
  alignScore: number;
  verdicts: Record<string, PartVerdict>;
};

export default function LocalLab() {
  const [refs, setRefs] = useState<Reference[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("overview");
  const [zone, setZone] = useState<Poly | null>(null);
  const [parts, setParts] = useState<Part[]>([]);
  const [name, setName] = useState("");
  const [capturing, setCapturing] = useState(false);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [sensitivity, setSensitivity] = useState(0.5);

  const active = useMemo(() => refs.find((r) => r.id === activeId) || null, [refs, activeId]);

  useEffect(() => { setRefs(loadReferences()); }, []);

  useEffect(() => {
    if (active) {
      setZone(active.zone);
      setParts(active.parts);
      setName(active.name);
    }
  }, [activeId]);

  async function onNewReference(file: File | null) {
    if (!file) return;
    setError("");
    setBusy("Procesando foto patrón…");
    try {
      const url = await fileToDataUrl(file, 1600);
      const img = new Image();
      await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = () => rej(new Error("imagen")); img.src = url; });
      const ref: Reference = {
        id: newId("ref"),
        name: `Referencia ${refs.length + 1}`,
        createdAt: new Date().toISOString(),
        imageDataUrl: url,
        width: img.naturalWidth,
        height: img.naturalHeight,
        zone: null,
        parts: [],
      };
      const { ok, error: saveErr, refs: next } = upsertReference(ref);
      if (!ok) setError(saveErr || "No se pudo guardar");
      setRefs(next);
      setActiveId(ref.id);
      setMode("annotate");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error al cargar la foto");
    } finally {
      setBusy("");
    }
  }

  function persist() {
    if (!active) return;
    const updated: Reference = { ...active, name, zone, parts };
    const { ok, error: saveErr, refs: next } = upsertReference(updated);
    if (!ok) { setError(saveErr || "No se pudo guardar"); return; }
    setRefs(next);
    setError("");
    setBusy("Guardado ✓");
    window.setTimeout(() => setBusy(""), 1200);
  }

  async function onInspectionCaptured(file: File) {
    setCapturing(false);
    if (!active) return;
    setBusy("Comparando piezas…");
    setError("");
    try {
      const inspUrl = await fileToDataUrl(file, 1280);
      const golden = await loadGray(active.imageDataUrl, 768);
      const insp = await loadGray(inspUrl, 768);
      const t = align(golden, insp, active.zone);
      const verdicts: Record<string, PartVerdict> = {};
      for (const part of active.parts) {
        const sig = partSignals(golden, insp, part.bbox, t);
        verdicts[part.id] = scorePart(sig, sensitivity);
      }
      setInspection({ inspUrl, alignScore: t.score, verdicts });
      setMode("results");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error al comparar");
    } finally {
      setBusy("");
    }
  }

  // Re-threshold instantly when sensitivity changes (no realignment needed).
  const verdictsForSensitivity = useMemo(() => {
    if (!inspection) return null;
    const out: Record<string, PartVerdict> = {};
    for (const [id, v] of Object.entries(inspection.verdicts)) out[id] = scorePart(v.signals, sensitivity);
    return out;
  }, [inspection, sensitivity]);

  const decisionMap = useMemo(() => {
    const map: Record<string, "present" | "missing" | "uncertain"> = {};
    const src = verdictsForSensitivity || {};
    for (const [id, v] of Object.entries(src)) map[id] = v.decision;
    return map;
  }, [verdictsForSensitivity]);

  const summary = useMemo(() => {
    const v = verdictsForSensitivity || {};
    const all = Object.values(v);
    return {
      missing: all.filter((x) => x.decision === "missing").length,
      uncertain: all.filter((x) => x.decision === "uncertain").length,
      present: all.filter((x) => x.decision === "present").length,
      total: all.length,
    };
  }, [verdictsForSensitivity]);

  return (
    <>
      <header className="topbar">
        <div>
          <h1>Prueba local</h1>
          <p>Registra una referencia, marca las piezas y captura con guía. Todo corre en el dispositivo, sin servidor.</p>
        </div>
        {active ? (
          <div className="labSteps">
            <button className={mode === "annotate" ? "on" : ""} onClick={() => setMode("annotate")} type="button">1 · Anotar</button>
            <button className={mode === "results" ? "on" : ""} onClick={() => active.parts.length && setCapturing(true)} type="button" disabled={!active.parts.length}>2 · Inspeccionar</button>
          </div>
        ) : null}
      </header>

      {error ? <p className="errorText inlineError">{error}</p> : null}
      {busy ? <p className="infoText inlineError">{busy}</p> : null}

      {mode === "overview" || !active ? (
        <section className="labOverview">
          <div className="panel labNewCard">
            <h2>Nueva referencia</h2>
            <p>Toma o sube una foto del montaje correcto (todas las piezas presentes).</p>
            <label className="labBigButton">
              <span>📷 Tomar / subir foto patrón</span>
              <input type="file" accept="image/*" capture="environment" onChange={(e) => onNewReference(e.target.files?.[0] ?? null)} />
            </label>
          </div>
          <div className="panel labRefList">
            <h2>Referencias guardadas</h2>
            {refs.length ? refs.map((r) => (
              <div key={r.id} className={`labRefRow ${activeId === r.id ? "active" : ""}`}>
                <img src={r.imageDataUrl} alt={r.name} onClick={() => { setActiveId(r.id); setMode("annotate"); }} />
                <div className="labRefMeta" onClick={() => { setActiveId(r.id); setMode("annotate"); }}>
                  <strong>{r.name}</strong>
                  <span>{r.parts.length} piezas · {r.zone ? "zona ✓" : "sin zona"}</span>
                </div>
                <button className="dangerButton" type="button" onClick={() => { setRefs(deleteReference(r.id)); if (activeId === r.id) { setActiveId(null); setMode("overview"); } }}>Borrar</button>
              </div>
            )) : <span className="emptyText compact">Aún no hay referencias.</span>}
          </div>
        </section>
      ) : null}

      {active && mode === "annotate" ? (
        <section className="labWork">
          <div className="labWorkBar">
            <input className="labLabelInput wide" value={name} onChange={(e) => setName(e.target.value)} placeholder="Nombre de la referencia" />
            <div className="labCounts">
              <span>{parts.length} piezas</span>
              <span>{zone ? "zona definida" : "define la zona"}</span>
            </div>
            <button className="secondary" type="button" onClick={() => { setActiveId(null); setMode("overview"); }}>← Referencias</button>
            <button className="primary" type="button" onClick={persist}>Guardar</button>
            <button className="primary" type="button" disabled={!parts.length} onClick={() => { persist(); setCapturing(true); }}>Inspeccionar →</button>
          </div>
          <AnnotationCanvas
            imageUrl={active.imageDataUrl}
            zone={zone}
            parts={parts}
            onZoneChange={setZone}
            onPartsChange={setParts}
          />
          <p className="labTip">Consejo: acerca con dos dedos o la rueda. La lupa muestra lo que está bajo tu dedo para marcar piezas diminutas con precisión.</p>
        </section>
      ) : null}

      {active && mode === "results" && inspection ? (
        <section className="labWork">
          <div className="labWorkBar">
            <div className={`labVerdict ${summary.missing ? "bad" : summary.uncertain ? "warn" : "good"}`}>
              {summary.missing ? `${summary.missing} faltante(s)` : summary.uncertain ? `${summary.uncertain} a revisar` : "Todo presente"}
            </div>
            <div className="labCounts">
              <span>alineación {Math.round(inspection.alignScore * 100)}%</span>
              <span>{summary.present}/{summary.total} ok</span>
            </div>
            <label className="labSens">Sensibilidad
              <input type="range" min={0} max={100} value={Math.round(sensitivity * 100)} onChange={(e) => setSensitivity(Number(e.target.value) / 100)} />
            </label>
            <button className="secondary" type="button" onClick={() => setMode("annotate")}>← Anotar</button>
            <button className="primary" type="button" onClick={() => setCapturing(true)}>Volver a inspeccionar</button>
          </div>
          <div className="labResultGrid">
            <div className="labResultCanvas">
              <span className="labResultTag">Referencia + veredicto</span>
              <AnnotationCanvas
                imageUrl={active.imageDataUrl}
                zone={active.zone}
                parts={active.parts}
                partVerdicts={decisionMap}
                onZoneChange={() => {}}
                onPartsChange={() => {}}
                readOnly
              />
            </div>
            <div className="labResultSide">
              <span className="labResultTag">Foto capturada</span>
              <img className="labResultPhoto" src={inspection.inspUrl} alt="captura" />
              <div className="labResultList">
                {active.parts.map((p) => {
                  const v = (verdictsForSensitivity || {})[p.id];
                  if (!v) return null;
                  return (
                    <div key={p.id} className={`labResultRow v-${v.decision}`}>
                      <strong>{p.label}</strong>
                      <span>{v.decision === "missing" ? "falta" : v.decision === "uncertain" ? "revisar" : "ok"}</span>
                      <small>cambio {Math.round(v.change * 100)}%</small>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </section>
      ) : null}

      {capturing && active ? (
        <GuidedCapture
          referenceUrl={active.imageDataUrl}
          zone={active.zone || zone}
          onCapture={onInspectionCaptured}
          onCancel={() => setCapturing(false)}
        />
      ) : null}
    </>
  );
}
