import React, { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Check, ChevronRight, CopyCheck, FolderPlus, ImagePlus, Layers3, PanelRightClose, PanelRightOpen, Plus, RotateCcw, ScanLine, Upload } from "lucide-react";
import { getJson, postJson, uploadSystemFile, messageFrom } from "../utils/api";
import AnnotatorScreen from "./AnnotatorScreen";
import AnnotateCanvas from "./AnnotateCanvas";
import { type AnnElement, type Category, bboxOfPolygon, newId, rectPolygon, slugify } from "./types";
import type { Point } from "../utils/geometry";
import "./annotate.css";

const CATEGORY_PALETTE = ["#2563eb", "#0891b2", "#7c3aed", "#db2777", "#16a34a", "#ea580c", "#0d9488", "#9333ea"];
const CATEGORY_STORE = "moldVision.annotationCategories";
const PROJECT_STORE = "moldVision.annotationProject";

type ImageSlot = "reference" | "compareA" | "compareB";
type Mode = "setup" | "edit" | "mapping" | "review";

type SectionAsset = {
  file?: File | null;
  url: string;
  uri?: string | null;
};

type MoldSectionDraft = {
  id: string;
  name: string;
  reference?: SectionAsset;
  compareA?: SectionAsset;
  compareB?: SectionAsset;
};

type MoldProject = {
  id: string;
  name: string;
  sectionCount: number;
  sections: MoldSectionDraft[];
};

type ReviewDraft = {
  sectionId: string;
  slot: "compareA" | "compareB";
  elements: AnnElement[];
  approved: boolean;
  source: "alignment" | "auto_draft" | "copy";
  alignmentConfidence: number;
  message: string;
};

type Props = {
  onExit: () => void;
  initialMoldName?: string;
};

export default function AnnotateApp({ onExit, initialMoldName }: Props) {
  const [categories, setCategories] = useState<Category[]>([]);
  const [creatingCategory, setCreatingCategory] = useState(false);
  const [project, setProject] = useState<MoldProject>(() => initialProject());
  const [activeSectionId, setActiveSectionId] = useState(project.sections[0]?.id ?? "section_01");
  const [activeSlot, setActiveSlot] = useState<ImageSlot>("reference");
  const [mode, setMode] = useState<Mode>("setup");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [annotations, setAnnotations] = useState<Record<string, AnnElement[]>>({});
  const [evaluationZones, setEvaluationZones] = useState<Record<string, Point[]>>({});
  const [reviewDrafts, setReviewDrafts] = useState<ReviewDraft[]>([]);
  const [rightRailOpen, setRightRailOpen] = useState(true);

  useEffect(() => { void loadCategories(); }, []);
  useEffect(() => persistProject(project), [project]);
  useEffect(() => {
    if (initialMoldName) setProject((current) => ({ ...current, name: initialMoldName }));
  }, [initialMoldName]);

  const activeSection = project.sections.find((section) => section.id === activeSectionId) ?? project.sections[0];
  const activeAsset = activeSection?.[activeSlot];
  const canAnnotate = project.sections.some((section) => section.reference?.url);
  const annotationKey = keyFor(activeSection?.id ?? "", activeSlot);
  const evaluationZone = activeSection ? evaluationZones[activeSection.id] ?? null : null;
  const sectionReferenceCount = annotations[keyFor(activeSection?.id ?? "", "reference")]?.length ?? 0;
  const allAnnotatedZonesReady = project.sections.every((section) => {
    const hasReferenceAnnotations = (annotations[keyFor(section.id, "reference")] ?? []).length > 0;
    return !hasReferenceAnnotations || (evaluationZones[section.id]?.length ?? 0) >= 3;
  });
  const canVerify = project.sections.some((section) => {
    const hasReferenceAnnotations = (annotations[keyFor(section.id, "reference")] ?? []).length > 0;
    const hasTargets = (["compareA", "compareB"] as const).some((slot) => Boolean(section[slot]?.url));
    return hasReferenceAnnotations && hasTargets;
  }) && allAnnotatedZonesReady;

  const totals = useMemo(() => {
    const all = Object.values(annotations).flat();
    const mapped = reviewDrafts.reduce((sum, draft) => sum + draft.elements.length, 0);
    return { annotated: all.length, mapped, categories: categories.length };
  }, [annotations, reviewDrafts, categories.length]);

  async function loadCategories() {
    const local = readCategories();
    if (local.length) setCategories(local);
    try {
      const records = await getJson("/v1/categories");
      const parsed = Array.isArray(records) ? records.map(toCategory).filter(Boolean) as Category[] : [];
      if (parsed.length) {
        setCategories(parsed);
        writeCategories(parsed);
      }
    } catch {
      // Local registry remains usable when the backend is offline.
    }
  }

  async function createCategory(name: string): Promise<Category | null> {
    const existing = categories.find((c) => c.name.toLowerCase() === name.toLowerCase());
    if (existing) return existing;
    const category: Category = {
      id: newId("cat"),
      name,
      slug: slugify(name),
      color: CATEGORY_PALETTE[categories.length % CATEGORY_PALETTE.length],
    };
    const next = [...categories, category];
    setCreatingCategory(true);
    setCategories(next);
    writeCategories(next);
    try {
      await postJson("/v1/categories", category);
    } catch {
      // Keep local category; persistence can sync later.
    } finally {
      setCreatingCategory(false);
    }
    return category;
  }

  function updateProject(patch: Partial<MoldProject>) {
    setProject((current) => ({ ...current, ...patch }));
  }

  function setSectionCount(count: number) {
    const nextCount = Math.max(1, Math.min(16, count));
    setProject((current) => {
      const sections = Array.from({ length: nextCount }, (_, index) => current.sections[index] ?? {
        id: `section_${String(index + 1).padStart(2, "0")}`,
        name: `Sección ${index + 1}`,
      });
      if (!sections.some((section) => section.id === activeSectionId)) setActiveSectionId(sections[0].id);
      return { ...current, sectionCount: nextCount, sections };
    });
  }

  function setSectionAsset(sectionId: string, slot: ImageSlot, file: File | null) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setProject((current) => ({
      ...current,
      sections: current.sections.map((section) => section.id === sectionId ? { ...section, [slot]: { file, url, uri: null } } : section),
    }));
  }

  function startAnnotating() {
    const firstReady = project.sections.find((section) => section.reference?.url);
    if (!firstReady) return;
    setActiveSectionId(firstReady.id);
    setActiveSlot("reference");
    setMode("edit");
    setNotice("");
    setError("");
  }

  async function save(elements: AnnElement[]) {
    if (!activeSection || !activeAsset) return;
    setSaving(true);
    setError("");
    setNotice("");
    try {
      const activeDraft = activeSlot === "reference"
        ? null
        : reviewDrafts.find((draft) => draft.sectionId === activeSection.id && draft.slot === activeSlot && !draft.approved);
      if (activeDraft) {
        setAnnotations((current) => ({ ...current, [annotationKey]: elements }));
        setReviewDrafts((current) => current.map((draft) => (
          draft.sectionId === activeSection.id && draft.slot === activeSlot
            ? { ...draft, elements, approved: false, message: "Editado, pendiente de aprobar." }
            : draft
        )));
        setNotice(`Borrador actualizado: ${elements.length} pieza(s).`);
        setMode("review");
        return;
      }

      let uri = activeAsset.uri;
      if (!uri && activeAsset.file) {
        uri = await uploadSystemFile(activeAsset.file, project.id, activeSection.id, "annotation");
        patchAssetUri(activeSection.id, activeSlot, uri);
      }
      if (uri) {
        if (activeSlot === "reference") {
          await postJson(`/v1/zones/${encodeURIComponent(activeSection.id)}/reference`, {
            family: project.id,
            image_uri: uri,
            reference_id: "golden_sample",
          }).catch(() => undefined);
        }
        await postJson("/v1/annotations", {
          image_uri: uri,
          family: project.id,
          zone_id: activeSection.id,
          split: activeSlot === "reference" ? "train" : "val",
          annotations: elements.map(toPayload),
          metadata: {
            mold_name: project.name,
            image_slot: activeSlot,
            purpose: "future_auto_label_training",
            evaluation_zone: evaluationZone?.map((p) => [p.x, p.y]) ?? null,
          },
        });
      }
      setAnnotations((current) => ({ ...current, [annotationKey]: elements }));
      setNotice(`Guardado: ${elements.length} pieza(s).`);
      if (activeSlot === "reference") {
        const drafts = await mapSection(activeSection.id, elements);
        if (drafts.length) {
          upsertReviewDrafts(drafts);
          setNotice(`Guardado. ${drafts.length} golden(s) auto-anotado(s) para revisar.`);
          setMode("review");
        }
      }
    } catch (e) {
      setError(messageFrom(e));
    } finally {
      setSaving(false);
    }
  }

  function patchAssetUri(sectionId: string, slot: ImageSlot, uri: string) {
    setProject((current) => ({
      ...current,
      sections: current.sections.map((section) => {
        const asset = section[slot];
        return section.id === sectionId && asset ? { ...section, [slot]: { ...asset, uri } } : section;
      }),
    }));
  }

  // Upload an asset once and cache its object URI (needed before the backend can
  // align/transfer or persist annotations against it).
  async function ensureAssetUri(sectionId: string, slot: ImageSlot): Promise<string | null> {
    const section = project.sections.find((s) => s.id === sectionId);
    const asset = section?.[slot];
    if (!asset) return null;
    if (asset.uri) return asset.uri;
    if (!asset.file) return null;
    const uri = await uploadSystemFile(asset.file, project.id, sectionId, "annotation");
    patchAssetUri(sectionId, slot, uri);
    return uri;
  }

  async function verify() {
    if (!canVerify) {
      setNotice("Selecciona la zona azul de evaluación antes de continuar.");
      return;
    }
    setMode("mapping");
    setNotice("Mapeando anotaciones");
    setError("");
    const drafts: ReviewDraft[] = [];
    for (const section of project.sections) {
      const source = annotations[keyFor(section.id, "reference")] ?? [];
      if (!source.length) continue;
      drafts.push(...await mapSection(section.id, source));
    }
    setReviewDrafts(drafts);
    setMode("review");
    setNotice("");
  }

  async function mapSection(sectionId: string, source: AnnElement[]): Promise<ReviewDraft[]> {
    const section = project.sections.find((item) => item.id === sectionId);
    if (!section?.reference?.url || !source.length) return [];
    const slots = (["compareA", "compareB"] as const).filter((slot) => section[slot]?.url);
    if (!slots.length) return [];

    const referenceUri = await ensureAssetUri(section.id, "reference");
    const targets: Array<{ slot: "compareA" | "compareB"; uri: string }> = [];
    for (const slot of slots) {
      const uri = await ensureAssetUri(section.id, slot);
      if (uri) targets.push({ slot, uri });
    }
    if (!referenceUri || !targets.length) return [];

    const transferByUri = new Map<string, any>();
    try {
      const response = await postJson("/v1/annotations/transfer", {
        reference_image_uri: referenceUri,
        target_image_uris: targets.map((target) => target.uri),
        annotations: source.map(toPayload),
      });
      const results = Array.isArray(response?.results) ? response.results : [];
      for (const result of results) {
        if (result?.image_uri) transferByUri.set(String(result.image_uri), result);
      }
    } catch {
      transferByUri.clear();
    }

    const drafts: ReviewDraft[] = [];
    for (const target of targets) {
      const transfer = transferByUri.get(target.uri);
      const ok = Boolean(transfer?.ok && Array.isArray(transfer?.annotations) && transfer.annotations.length);
      if (ok) {
        const confidence = Number(transfer.alignment_confidence ?? 0);
        const note = `Mapeado (alineación ${Math.round(confidence * 100)}%)`;
        drafts.push({
          sectionId,
          slot: target.slot,
          approved: false,
          elements: transfer.annotations.map((payload: unknown) => fromPayload(payload, note)),
          source: "alignment",
          alignmentConfidence: confidence,
          message: note,
        });
        continue;
      }

      const autoDraft = await autoDraftForTarget(sectionId, target.uri);
      if (autoDraft.length) {
        drafts.push({
          sectionId,
          slot: target.slot,
          approved: false,
          elements: autoDraft,
          source: "auto_draft",
          alignmentConfidence: 0,
          message: "Borrador automático. Revisa antes de aprobar.",
        });
        continue;
      }

      drafts.push({
        sectionId,
        slot: target.slot,
        approved: false,
        elements: source.map((element) => ({ ...element, id: newId("map"), notes: element.notes || "Copia directa (sin alinear)" })),
        source: "copy",
        alignmentConfidence: 0,
        message: "Copia directa. Ajusta manualmente antes de aprobar.",
      });
    }
    return drafts;
  }

  async function autoDraftForTarget(sectionId: string, imageUri: string): Promise<AnnElement[]> {
    try {
      const response = await postJson("/v1/annotations/auto-draft", {
        family: project.id,
        zone_id: sectionId,
        image_uri: imageUri,
      }) as { annotations?: unknown[]; message?: string };
      const note = response.message || "Borrador automático";
      return Array.isArray(response.annotations) ? response.annotations.map((payload) => fromPayload(payload, note)) : [];
    } catch {
      return [];
    }
  }

  function upsertReviewDrafts(drafts: ReviewDraft[]) {
    setReviewDrafts((current) => {
      const next = current.filter((existing) => !drafts.some((draft) => draft.sectionId === existing.sectionId && draft.slot === existing.slot));
      return [...next, ...drafts];
    });
  }

  async function approveDraft(sectionId: string, slot: "compareA" | "compareB") {
    const draft = reviewDrafts.find((item) => item.sectionId === sectionId && item.slot === slot);
    if (!draft) return;
    setAnnotations((current) => ({ ...current, [keyFor(sectionId, slot)]: draft.elements }));
    setReviewDrafts((current) => current.map((item) => item.sectionId === sectionId && item.slot === slot ? { ...item, approved: true } : item));
    // Persist the mapped annotations as extra labeled training data (val split).
    try {
      const uri = await ensureAssetUri(sectionId, slot);
      if (uri) {
        await postJson("/v1/annotations", {
          image_uri: uri,
          family: project.id,
          zone_id: sectionId,
          split: "val",
          annotations: draft.elements.map(toPayload),
          metadata: {
            mold_name: project.name,
            image_slot: slot,
            purpose: "future_auto_label_training",
            mapped_from_reference: true,
            evaluation_zone: evaluationZones[sectionId]?.map((p) => [p.x, p.y]) ?? null,
          },
        });
      }
      setNotice(`Aprobado: ${draft.elements.length} pieza(s).`);
    } catch (e) {
      setError(messageFrom(e)); // keep the local approval even if persistence fails
    }
  }

  function discardDraft(sectionId: string, slot: "compareA" | "compareB") {
    setReviewDrafts((current) => current.filter((draft) => !(draft.sectionId === sectionId && draft.slot === slot)));
    setAnnotations((current) => {
      const next = { ...current };
      delete next[keyFor(sectionId, slot)];
      return next;
    });
  }

  function editDraft(sectionId: string, slot: "compareA" | "compareB") {
    const draft = reviewDrafts.find((item) => item.sectionId === sectionId && item.slot === slot);
    if (draft) setAnnotations((current) => ({ ...current, [keyFor(sectionId, slot)]: draft.elements }));
    setActiveSectionId(sectionId);
    setActiveSlot(slot);
    setMode("edit");
  }

  return (
    <div className={`annAppShell ${mode === "edit" ? "isEditing" : ""}`}>
      <header className="annTopFrame">
        <div className="annTopLeft">
          <button className="annIconButton" type="button" onClick={onExit} title="Volver">
            <ArrowLeft size={18} />
          </button>
          <div className="annTopMark"><Layers3 size={17} /></div>
          <div>
            <strong>{project.name || "Nuevo molde"}</strong>
            <span>{mode === "review" ? "Revisión de goldens auto-anotados" : "Anotación manual y asistida"}</span>
          </div>
        </div>
        <div className="annTopActions">
          {mode === "edit" && activeSection ? (
            <div className="annImageSwitch">
              {(["reference", "compareA", "compareB"] as const).map((slot) => (
                <button key={slot} type="button" disabled={!activeSection[slot]?.url} className={activeSlot === slot ? "on" : ""} onClick={() => setActiveSlot(slot)}>
                  {slotLabel(slot)}
                </button>
              ))}
            </div>
          ) : null}
          <button className="annTopButton" type="button" onClick={() => setMode("setup")}><FolderPlus size={14} /> Preparar</button>
          <button className="annTopButton primary" type="button" onClick={verify} disabled={!canVerify}><ScanLine size={14} /> Verificar</button>
          {mode === "edit" ? (
            <button className="annIconButton" type="button" onClick={() => setRightRailOpen((open) => !open)} title={rightRailOpen ? "Contraer panel" : "Expandir panel"}>
              {rightRailOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
            </button>
          ) : null}
        </div>
      </header>
      <aside className="annProjectRail">
        <div className="annRailHead">
          <span className="annRailIcon"><Layers3 size={16} /></span>
          <div>
            <strong>Moldes</strong>
            <small>Goldens y anotaciones</small>
          </div>
        </div>

        <div className="annStatGrid">
          <span><b>{project.sectionCount}</b> secc.</span>
          <span><b>{totals.annotated}</b> piezas</span>
          <span><b>{totals.categories}</b> cats.</span>
        </div>

        <div className="annRailSection">
          <label>Molde</label>
          <input value={project.name} onChange={(e) => updateProject({ name: e.target.value })} />
        </div>

        <div className="annRailSection">
          <label>Secciones</label>
          <div className="annStepper">
            <button type="button" onClick={() => setSectionCount(project.sectionCount - 1)}>-</button>
            <input type="number" min={1} max={16} value={project.sectionCount} onChange={(e) => setSectionCount(Number(e.target.value) || 1)} />
            <button type="button" onClick={() => setSectionCount(project.sectionCount + 1)}>+</button>
          </div>
        </div>

        <div className="annSectionList">
          {project.sections.map((section, index) => (
            <button
              type="button"
              key={section.id}
              className={section.id === activeSectionId ? "on" : ""}
              onClick={() => { setActiveSectionId(section.id); setActiveSlot("reference"); }}
            >
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{section.name}</strong>
              <small>{sectionReadyLabel(section)}</small>
            </button>
          ))}
        </div>

        <button className="annRailAction" type="button" disabled={!canAnnotate} onClick={startAnnotating}>
          <ChevronRight size={15} /> Anotar goldens
        </button>
      </aside>

      <section className="annMainPane">
        {error ? <p className="errorText inlineError">{error}</p> : null}
        {notice ? <p className="infoText inlineError">{notice}</p> : null}

        {mode === "setup" ? (
          <SetupBoard project={project} activeSectionId={activeSectionId} onSelectSection={setActiveSectionId} onFile={setSectionAsset} onStart={startAnnotating} canStart={canAnnotate} />
        ) : null}

        {mode === "edit" && activeSection && activeAsset ? (
          <div className="annEditStack">
            <AnnotatorScreen
              key={annotationKey}
              imageUrl={activeAsset.url}
              initialElements={annotations[annotationKey] ?? []}
              categories={categories}
              creatingCategory={creatingCategory}
              onCreateCategory={createCategory}
              onSave={save}
              saving={saving}
              onVerify={verify}
              verifyDisabledReason={!evaluationZone ? "Selecciona la zona azul de evaluación" : sectionReferenceCount ? undefined : "Anota la referencia antes de verificar"}
              evaluationZone={evaluationZone}
              onEvaluationZoneChange={(zone) => setEvaluationZones((current) => ({ ...current, [activeSection.id]: zone }))}
              rightRailOpen={rightRailOpen}
              onToggleRail={() => setRightRailOpen((open) => !open)}
            />
          </div>
        ) : null}

        {mode === "mapping" ? (
          <div className="annMappingState">
            <ScanLine size={28} />
            <strong>Mapeando anotaciones</strong>
            <span>Proyectando categorías y regiones sobre las dos imágenes de comparación.</span>
          </div>
        ) : null}

        {mode === "review" ? (
          <ReviewBoard project={project} drafts={reviewDrafts} onApprove={approveDraft} onDiscard={discardDraft} onEdit={editDraft} />
        ) : null}
      </section>
    </div>
  );
}

function SetupBoard({
  project,
  activeSectionId,
  onSelectSection,
  onFile,
  onStart,
  canStart,
}: {
  project: MoldProject;
  activeSectionId: string;
  onSelectSection: (id: string) => void;
  onFile: (sectionId: string, slot: ImageSlot, file: File | null) => void;
  onStart: () => void;
  canStart: boolean;
}) {
  const section = project.sections.find((item) => item.id === activeSectionId) ?? project.sections[0];
  return (
    <div className="annSetup">
      <div className="annSetupList">
        {project.sections.map((item, index) => (
          <button key={item.id} type="button" className={item.id === section.id ? "on" : ""} onClick={() => onSelectSection(item.id)}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <strong>{item.name}</strong>
            <small>{sectionReadyLabel(item)}</small>
          </button>
        ))}
      </div>
      <div className="annUploadGrid">
        <UploadTile title="Golden base" asset={section.reference} onFile={(file) => onFile(section.id, "reference", file)} />
        <UploadTile title="Golden 2" asset={section.compareA} onFile={(file) => onFile(section.id, "compareA", file)} />
        <UploadTile title="Golden 3" asset={section.compareB} onFile={(file) => onFile(section.id, "compareB", file)} />
      </div>
      <button className="annStartButton" type="button" disabled={!canStart} onClick={onStart}>
        <Upload size={16} /> Anotar goldens
      </button>
    </div>
  );
}

function UploadTile({ title, asset, onFile }: { title: string; asset?: SectionAsset; onFile: (file: File | null) => void }) {
  return (
    <label className="annUploadTile">
      <input type="file" accept="image/*" capture="environment" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
      {asset?.url ? <img src={asset.url} alt={title} /> : <ImagePlus size={24} />}
      <span>{title}</span>
      <small>{asset?.file?.name || "Cargar o tomar foto"}</small>
    </label>
  );
}

function SectionTabs({
  project,
  activeSectionId,
  activeSlot,
  onSection,
  onSlot,
}: {
  project: MoldProject;
  activeSectionId: string;
  activeSlot: ImageSlot;
  onSection: (id: string) => void;
  onSlot: (slot: ImageSlot) => void;
}) {
  const active = project.sections.find((section) => section.id === activeSectionId) ?? project.sections[0];
  return (
    <div className="annContextBar">
      <div className="annSectionTabs">
        {project.sections.map((section) => (
          <button key={section.id} type="button" className={section.id === activeSectionId ? "on" : ""} onClick={() => onSection(section.id)}>
            {section.name}
          </button>
        ))}
      </div>
      <div className="annImageTabs">
        {(["reference", "compareA", "compareB"] as const).map((slot) => (
          <button key={slot} type="button" disabled={!active?.[slot]?.url} className={activeSlot === slot ? "on" : ""} onClick={() => onSlot(slot)}>
            {slotLabel(slot)}
          </button>
        ))}
      </div>
    </div>
  );
}

function ReviewBoard({
  project,
  drafts,
  onApprove,
  onDiscard,
  onEdit,
}: {
  project: MoldProject;
  drafts: ReviewDraft[];
  onApprove: (sectionId: string, slot: "compareA" | "compareB") => void;
  onDiscard: (sectionId: string, slot: "compareA" | "compareB") => void;
  onEdit: (sectionId: string, slot: "compareA" | "compareB") => void;
}) {
  if (!drafts.length) {
    return (
      <div className="annMappingState">
        <RotateCcw size={28} />
        <strong>Sin mapas listos</strong>
        <span>Vuelve a editar una referencia con anotaciones y verifica otra vez.</span>
      </div>
    );
  }

  return (
    <div className="annReviewGrid">
      {drafts.map((draft) => {
        const section = project.sections.find((item) => item.id === draft.sectionId);
        const asset = section?.[draft.slot];
        if (!section || !asset?.url) return null;
        return (
          <article key={`${draft.sectionId}_${draft.slot}`} className={draft.approved ? "annReviewCard approved" : "annReviewCard"}>
            <div className="annReviewHead">
              <div>
                <strong>{section.name} · {slotLabel(draft.slot)}</strong>
                <span>{draft.elements.length} anotaciones · {draftLabel(draft)}</span>
              </div>
              {draft.approved ? <span className="annApproved"><Check size={13} /> Aprobado</span> : null}
            </div>
            <div className="annReviewCanvas">
              <AnnotateCanvas
                imageUrl={asset.url}
                elements={draft.elements}
                selectedId={null}
                highlightCategoryId={null}
                tool="select"
                importance="relevant"
                category={null}
                onChange={() => undefined}
                onSelect={() => undefined}
                readOnly
              />
            </div>
            <div className="annReviewActions">
              <button type="button" onClick={() => onEdit(draft.sectionId, draft.slot)}><Plus size={13} /> Editar</button>
              <button type="button" onClick={() => onDiscard(draft.sectionId, draft.slot)}><RotateCcw size={13} /> Descartar</button>
              <button type="button" className="primary" onClick={() => onApprove(draft.sectionId, draft.slot)} disabled={draft.approved}><CopyCheck size={13} /> Aprobar</button>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function draftLabel(draft: ReviewDraft): string {
  if (draft.source === "alignment") return `alineación ${Math.round(draft.alignmentConfidence * 100)}%`;
  if (draft.source === "auto_draft") return "auto";
  return "copia";
}

function initialProject(): MoldProject {
  const stored = readProject();
  if (stored) return stored;
  return {
    id: newId("mold"),
    name: "Nuevo molde",
    sectionCount: 3,
    sections: [
      { id: "section_01", name: "Sección 1" },
      { id: "section_02", name: "Sección 2" },
      { id: "section_03", name: "Sección 3" },
    ],
  };
}

function readProject(): MoldProject | null {
  try {
    const raw = localStorage.getItem(PROJECT_STORE);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as MoldProject;
    return parsed?.sections?.length ? parsed : null;
  } catch {
    return null;
  }
}

function persistProject(project: MoldProject) {
  const safe = {
    ...project,
    sections: project.sections.map((section) => ({
      ...section,
      reference: section.reference ? { url: section.reference.url, uri: section.reference.uri, file: null } : undefined,
      compareA: section.compareA ? { url: section.compareA.url, uri: section.compareA.uri, file: null } : undefined,
      compareB: section.compareB ? { url: section.compareB.url, uri: section.compareB.uri, file: null } : undefined,
    })),
  };
  try { localStorage.setItem(PROJECT_STORE, JSON.stringify(safe)); } catch { /* ignore quota */ }
}

function readCategories(): Category[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(CATEGORY_STORE) || "[]");
    return Array.isArray(parsed) ? parsed.filter((item) => item?.id && item?.name) : [];
  } catch {
    return [];
  }
}

function writeCategories(categories: Category[]) {
  try { localStorage.setItem(CATEGORY_STORE, JSON.stringify(categories)); } catch { /* ignore quota */ }
}

function keyFor(sectionId: string, slot: ImageSlot): string {
  return `${sectionId}:${slot}`;
}

function slotLabel(slot: ImageSlot): string {
  if (slot === "reference") return "Golden base";
  if (slot === "compareA") return "Golden 2";
  return "Golden 3";
}

function sectionReadyLabel(section: MoldSectionDraft): string {
  const count = [section.reference, section.compareA, section.compareB].filter((asset) => asset?.url).length;
  return `${count}/3 goldens`;
}

function toCategory(record: unknown): Category | null {
  if (!record || typeof record !== "object") return null;
  const r = record as Record<string, unknown>;
  const data = (r.data && typeof r.data === "object" ? r.data : r) as Record<string, unknown>;
  const name = String(data.name || r.name || "").trim();
  if (!name) return null;
  return {
    id: String(r.id || data.id || newId("cat")),
    name,
    slug: String(data.slug || slugify(name)),
    color: data.color ? String(data.color) : undefined,
  };
}

function toPayload(el: AnnElement) {
  return {
    id: el.id,
    element_id: el.id,
    class_name: el.categoryName || "pieza",
    bbox: el.bbox,
    status: "present" as const,
    shape: el.shape,
    polygon: el.polygon.map((p) => [p.x, p.y]),
    category_id: el.categoryId,
    category_name: el.categoryName,
    importance: el.importance,
    notes: el.notes,
  };
}

// Build an editor element from a backend annotation payload (e.g. a transfer
// result whose polygon has been warped onto the comparison image).
function fromPayload(payload: unknown, note?: string): AnnElement {
  const p = (payload && typeof payload === "object" ? payload : {}) as Record<string, any>;
  const polygon = Array.isArray(p.polygon) && p.polygon.length
    ? p.polygon.map((pt: [number, number]) => ({ x: pt[0], y: pt[1] }))
    : rectPolygon(...(Array.isArray(p.bbox) && p.bbox.length === 4 ? p.bbox : [0, 0, 0, 0]) as [number, number, number, number]);
  const importance = p.importance === "critical" || p.importance === "minor" ? p.importance : "relevant";
  return {
    id: newId("map"),
    shape: p.shape === "rect" ? "rect" : "polygon",
    polygon,
    bbox: Array.isArray(p.bbox) && p.bbox.length === 4 ? p.bbox as AnnElement["bbox"] : bboxOfPolygon(polygon),
    categoryId: p.category_id ?? null,
    categoryName: p.category_name ?? null,
    importance,
    notes: note ?? (typeof p.notes === "string" ? p.notes : undefined),
  };
}
