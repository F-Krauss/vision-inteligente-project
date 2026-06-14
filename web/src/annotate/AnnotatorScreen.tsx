// The annotation editor: toolbar (tools + importance + undo/redo + save +
// verify) wrapping the canvas, with the category registry and element inspector
// in a side rail. Owns the editing session state (tool, importance, active
// category, selection, and the undo/redo history of elements).

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Circle, Focus, Hand, MousePointer2, PanelRightOpen, Pentagon, Redo2, Save, ScanLine, SquareDashedMousePointer, Undo2 } from "lucide-react";
import type { Point } from "../utils/geometry";
import AnnotateCanvas from "./AnnotateCanvas";
import CategoryPanel from "./CategoryPanel";
import ElementInspector from "./ElementInspector";
import { type AnnElement, type Category, type Importance, type Tool, IMPORTANCES } from "./types";

type Props = {
  imageUrl: string;
  initialElements: AnnElement[];
  categories: Category[];
  creatingCategory?: boolean;
  onCreateCategory: (name: string) => Promise<Category | null> | void;
  onSave: (elements: AnnElement[]) => Promise<void> | void;
  saving?: boolean;
  onVerify?: (elements: AnnElement[]) => void;
  verifyDisabledReason?: string;
  evaluationZone: Point[] | null;
  onEvaluationZoneChange: (zone: Point[]) => void;
  rightRailOpen?: boolean;
  onToggleRail?: () => void;
};

export default function AnnotatorScreen({
  imageUrl,
  initialElements,
  categories,
  creatingCategory = false,
  onCreateCategory,
  onSave,
  saving = false,
  onVerify,
  verifyDisabledReason,
  evaluationZone,
  onEvaluationZoneChange,
  rightRailOpen = true,
  onToggleRail,
}: Props) {
  // Undo/redo history over the element list.
  const [history, setHistory] = useState<{ past: AnnElement[][]; present: AnnElement[]; future: AnnElement[][] }>(
    { past: [], present: initialElements, future: [] }
  );
  const elements = history.present;
  const baselineRef = useRef(initialElements);

  const [tool, setTool] = useState<Tool>(evaluationZone?.length ? "polygon" : "zone");
  const [importance, setImportance] = useState<Importance>("relevant");
  const [activeCategory, setActiveCategory] = useState<Category | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [zoneDirty, setZoneDirty] = useState(false);

  const commit = useCallback((next: AnnElement[]) => {
    setHistory((h) => ({ past: [...h.past, h.present], present: next, future: [] }));
  }, []);

  const undo = useCallback(() => {
    setHistory((h) => h.past.length ? { past: h.past.slice(0, -1), present: h.past[h.past.length - 1], future: [h.present, ...h.future] } : h);
  }, []);
  const redo = useCallback(() => {
    setHistory((h) => h.future.length ? { past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) } : h);
  }, []);

  const dirty = elements !== baselineRef.current;
  const selected = useMemo(() => elements.find((el) => el.id === selectedId) ?? null, [elements, selectedId]);

  function patchSelected(patch: Partial<AnnElement>) {
    if (!selected) return;
    commit(elements.map((el) => (el.id === selected.id ? { ...el, ...patch } : el)));
  }
  function deleteSelected() {
    if (!selected) return;
    commit(elements.filter((el) => el.id !== selected.id));
    setSelectedId(null);
  }

  async function handleCreateCategory(name: string) {
    const created = await onCreateCategory(name);
    if (created) setActiveCategory(created);
  }

  async function handleSave() {
    await onSave(elements);
    baselineRef.current = elements;
    setZoneDirty(false);
    // Touch history so `dirty` recomputes against the new baseline.
    setHistory((h) => ({ ...h }));
  }

  const counts = useMemo(() => {
    const c: Record<Importance, number> = { critical: 0, relevant: 0, minor: 0 };
    for (const el of elements) c[el.importance] += 1;
    return c;
  }, [elements]);

  const hasEvaluationZone = Boolean(evaluationZone?.length && evaluationZone.length >= 3);

  useEffect(() => {
    if (!hasEvaluationZone) setTool("zone");
  }, [hasEvaluationZone]);

  function setZone(zone: Point[]) {
    onEvaluationZoneChange(zone);
    setZoneDirty(true);
    setTool("polygon");
  }

  const canSave = hasEvaluationZone && (dirty || zoneDirty);

  function handleKeyDown(e: React.KeyboardEvent) {
    const tag = (document.activeElement as HTMLElement)?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if ((e.key === "Delete" || e.key === "Backspace") && selected) {
      e.preventDefault();
      deleteSelected();
    }
    if (e.key === "z" && (e.metaKey || e.ctrlKey) && !e.shiftKey) { e.preventDefault(); undo(); }
    if ((e.key === "z" && (e.metaKey || e.ctrlKey) && e.shiftKey) || (e.key === "y" && e.ctrlKey)) { e.preventDefault(); redo(); }
  }

  return (
    <div className="annEditor" tabIndex={-1} onKeyDown={handleKeyDown} style={{ outline: "none" }}>
      <div className="annToolbar">
        <div className="annToolGroup">
          <button className={tool === "zone" ? "on zoneOn" : hasEvaluationZone ? "zoneReady" : "zoneMissing"} onClick={() => setTool("zone")} type="button" title="Zona de evaluación obligatoria"><Focus size={14} /> Zona</button>
          <button className={tool === "polygon" ? "on" : ""} onClick={() => setTool("polygon")} disabled={!hasEvaluationZone} type="button" title="Polígono (clic para vértices)"><Pentagon size={14} /> Polígono</button>
          <button className={tool === "circle" ? "on" : ""} onClick={() => setTool("circle")} disabled={!hasEvaluationZone} type="button" title="Círculo (clic centro, clic radio)"><Circle size={14} /> Círculo</button>
          <button className={tool === "rect" ? "on" : ""} onClick={() => setTool("rect")} disabled={!hasEvaluationZone} type="button" title="Rectángulo (arrastrar)"><SquareDashedMousePointer size={14} /> Rectángulo</button>
          <button className={tool === "select" ? "on" : ""} onClick={() => setTool("select")} type="button" title="Editar / mover"><MousePointer2 size={14} /> Editar</button>
          <button className={tool === "pan" ? "on" : ""} onClick={() => setTool("pan")} type="button" title="Mover lienzo"><Hand size={14} /> Mover</button>
        </div>

        <div className="annToolGroup annImpToolbar">
          {IMPORTANCES.map((imp) => (
            <button
              key={imp.value}
              type="button"
              className={`annImpBtn ${importance === imp.value ? "on" : ""}`}
              style={importance === imp.value ? { borderColor: imp.color, boxShadow: `0 0 0 2px ${imp.color}33` } : undefined}
              onClick={() => setImportance(imp.value)}
              title={`Marcar piezas nuevas como ${imp.label}`}
            >
              <span className="annImpDot" style={{ background: imp.color }} />{imp.label}
            </button>
          ))}
        </div>

        <div className="annToolGroup">
          <button onClick={undo} disabled={!history.past.length} type="button" title="Deshacer"><Undo2 size={14} /></button>
          <button onClick={redo} disabled={!history.future.length} type="button" title="Rehacer"><Redo2 size={14} /></button>
        </div>

        <div className="annToolGroup grow annCountChips">
          <span>{elements.length} piezas</span>
          <span className="chip crit">{counts.critical}</span>
          <span className="chip rel">{counts.relevant}</span>
          <span className="chip min">{counts.minor}</span>
        </div>

        <div className="annToolGroup">
          <button className="primary" onClick={handleSave} disabled={saving || !canSave} type="button"><Save size={14} />{saving ? "Guardando..." : "Guardar"}</button>
          {onVerify ? (
            <button
              className="secondary"
              onClick={() => onVerify(elements)}
              disabled={Boolean(verifyDisabledReason) || !elements.length || !hasEvaluationZone}
              type="button"
              title={!hasEvaluationZone ? "Primero selecciona zona de evaluación" : verifyDisabledReason || "Mapear anotaciones a las imágenes de comparación"}
            ><ScanLine size={14} />Verificar</button>
          ) : null}
        </div>
      </div>

      <div className={`annBody ${rightRailOpen ? "" : "railClosed"}`}>
        <AnnotateCanvas
          imageUrl={imageUrl}
          elements={elements}
          selectedId={selectedId}
          highlightCategoryId={activeCategory?.id ?? null}
          tool={tool}
          importance={importance}
          category={activeCategory}
          evaluationZone={evaluationZone}
          onChange={commit}
          onEvaluationZoneChange={setZone}
          onSelect={setSelectedId}
        />
        <aside className={`annRail ${rightRailOpen ? "" : "collapsed"}`}>
          {rightRailOpen ? (
            <>
              <CategoryPanel
                categories={categories}
                elements={elements}
                activeCategoryId={activeCategory?.id ?? null}
                onSelectCategory={setActiveCategory}
                onCreateCategory={handleCreateCategory}
                busy={creatingCategory}
              />
              <ElementInspector
                element={selected}
                categories={categories}
                onChange={patchSelected}
                onDelete={deleteSelected}
              />
            </>
          ) : (
            <button className="annRailCollapsedButton" type="button" onClick={onToggleRail} title="Expandir panel derecho">
              <PanelRightOpen size={18} />
              <span>{categories.length}</span>
            </button>
          )}
        </aside>
      </div>
      <div className="annStatusBar">
        <span>Anotaciones: {elements.length}</span>
        <span className={hasEvaluationZone ? "zoneOk" : "zoneNeed"}>Zona azul: {hasEvaluationZone ? "lista" : "pendiente"}</span>
        <span>Categoria: {activeCategory?.name ?? "Sin categoria"}</span>
        <span>Importancia: {IMPORTANCES.map((imp) => <b key={imp.value}><i style={{ background: imp.color }} />{imp.label}</b>)}</span>
        <span>{dirty || zoneDirty ? "Cambios sin guardar" : "Guardado"}</span>
      </div>
    </div>
  );
}
