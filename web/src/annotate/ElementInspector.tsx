// Inspector for the selected element: shows importance, category, region
// (coordinates), area and id — and lets the operator edit importance, category
// and notes, or delete the element.

import React from "react";
import { type AnnElement, type Category, type Importance, IMPORTANCES, polygonArea } from "./types";

type Props = {
  element: AnnElement | null;
  categories: Category[];
  onChange: (patch: Partial<AnnElement>) => void;
  onDelete: () => void;
};

export default function ElementInspector({ element, categories, onChange, onDelete }: Props) {
  if (!element) {
    return (
      <div className="annPanel annInspector">
        <h3>Pieza</h3>
        <p className="annHint">Selecciona una pieza para ver y editar su información.</p>
      </div>
    );
  }

  const [x1, y1, x2, y2] = element.bbox;
  const areaPct = (polygonArea(element.polygon) * 100).toFixed(2);
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  return (
    <div className="annPanel annInspector">
      <h3>Pieza</h3>

      <div className="annField">
        <label>Importancia</label>
        <div className="annImpPicker">
          {IMPORTANCES.map((imp) => (
            <button
              key={imp.value}
              type="button"
              className={`annImpBtn ${element.importance === imp.value ? "on" : ""}`}
              style={element.importance === imp.value ? { borderColor: imp.color, boxShadow: `0 0 0 2px ${imp.color}33` } : undefined}
              onClick={() => onChange({ importance: imp.value as Importance })}
            >
              <span className="annImpDot" style={{ background: imp.color }} />{imp.label}
            </button>
          ))}
        </div>
      </div>

      <div className="annField">
        <label>Categoría</label>
        <select
          value={element.categoryId ?? ""}
          onChange={(e) => {
            const cat = categories.find((c) => c.id === e.target.value) || null;
            onChange({ categoryId: cat?.id ?? null, categoryName: cat?.name ?? null });
          }}
        >
          <option value="">Sin categoría</option>
          {categories.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
      </div>

      <dl className="annMeta">
        <div><dt>ID</dt><dd>{element.id}</dd></div>
        <div><dt>Forma</dt><dd>{element.shape === "rect" ? "Rectángulo" : "Polígono"} · {element.polygon.length} pts</dd></div>
        <div><dt>Región</dt><dd>{pct(x1)}, {pct(y1)} → {pct(x2)}, {pct(y2)}</dd></div>
        <div><dt>Área</dt><dd>{areaPct}%</dd></div>
      </dl>

      <div className="annField">
        <label>Notas</label>
        <textarea value={element.notes ?? ""} onChange={(e) => onChange({ notes: e.target.value })} rows={2} placeholder="Opcional" />
      </div>

      <button type="button" className="annDanger" onClick={onDelete}>Borrar pieza</button>
    </div>
  );
}
