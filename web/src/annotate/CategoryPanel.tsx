// Global category registry sidebar. Categories (screws, pistons, …) persist
// across molds; clicking one makes it the active tag for new shapes AND
// highlights every element already in that category.

import React, { useState } from "react";
import type { AnnElement, Category } from "./types";

type Props = {
  categories: Category[];
  elements: AnnElement[];
  activeCategoryId: string | null;
  onSelectCategory: (category: Category | null) => void;
  onCreateCategory: (name: string) => void;
  busy?: boolean;
};

export default function CategoryPanel({
  categories,
  elements,
  activeCategoryId,
  onSelectCategory,
  onCreateCategory,
  busy = false,
}: Props) {
  const [name, setName] = useState("");

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    onCreateCategory(trimmed);
    setName("");
  }

  const countFor = (id: string | null) => elements.filter((el) => el.categoryId === id).length;

  return (
    <div className="annPanel annCategories">
      <h3>Categorías</h3>
      <p className="annHint">Toca una para etiquetar piezas nuevas y resaltar las que ya pertenecen.</p>
      <div className="annCatList">
        <button
          type="button"
          className={`annCatRow ${activeCategoryId === null ? "on" : ""}`}
          onClick={() => onSelectCategory(null)}
        >
          <span className="annCatDot none" />
          <span className="annCatName">Sin categoría</span>
          <span className="annCatCount">{countFor(null)}</span>
        </button>
        {categories.map((cat) => (
          <button
            key={cat.id}
            type="button"
            className={`annCatRow ${activeCategoryId === cat.id ? "on" : ""}`}
            onClick={() => onSelectCategory(cat)}
          >
            <span className="annCatDot" style={{ background: cat.color || "#64748b" }} />
            <span className="annCatName">{cat.name}</span>
            <span className="annCatCount">{countFor(cat.id)}</span>
          </button>
        ))}
        {!categories.length ? <span className="annEmpty">Aún no hay categorías.</span> : null}
      </div>
      <form className="annCatCreate" onSubmit={submit}>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Nueva categoría (p. ej. Tornillo)" disabled={busy} />
        <button type="submit" disabled={busy || !name.trim()}>+ Crear</button>
      </form>
    </div>
  );
}
