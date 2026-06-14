import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Camera,
  CircleAlert,
  ClipboardCheck,
  Database,
  Gauge,
  LayoutDashboard,
  LogOut,
  PencilRuler,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  TrendingUp,
  TriangleAlert,
  UserCircle
} from "lucide-react";
import { supabase, supabaseConfigured } from "./utils/supabase";
import {
  API_BASE,
  isLocalRuntime,
  resolveUrl,
  displayUrl,
  postJson,
  getJson,
  uploadSystemFile,
  uploadFiles,
  messageFrom,
} from "./utils/api";
import { type Point, clamp, polygonPoints, distanceToSegment, insertPointInClosestSegment } from "./utils/geometry";
import AnnotateApp from "./annotate/AnnotateApp";
import "./styles.css";

type InspectionStatus = "idle" | "uploading" | "running" | "correct" | "review" | "retake_photo" | "error";
type View = "capture" | "molds" | "validations";

type Zone = {
  id: string;
  family: string;
  name?: string;
  label?: string;
};

type ZoneReference = {
  id: string;
  family: string;
  zone_id: string;
  reference_id: string;
  image_uri: string;
  image_url?: string;
  mask_uri?: string | null;
  mask_url?: string | null;
  tolerance?: Record<string, number>;
};

type ExpectedPiece = {
  id: string;
  class_name: string;
  name?: string | null;
  roi?: number[] | null;
  required?: boolean;
  critical?: boolean;
};

type AnnotationBox = {
  id: string;
  element_id?: string | null;
  class_name: string;
  bbox: [number, number, number, number];
  status: "present" | "missing" | "uncertain";
  notes?: string | null;
};

type AnnotatableImage = {
  id: string;
  image_uri: string;
  image_url?: string;
  family: string;
  zone_id: string;
  mold_id?: string | null;
  session_id?: string | null;
  created_at?: string;
};

type ResourceRecord = {
  id: string;
  created_at?: string;
  updated_at?: string;
  data?: Record<string, unknown>;
  [key: string]: unknown;
};

type MoldSummary = {
  id: string;
  name: string;
  family: string;
  zoneId: string;
  status: "ready" | "training" | "needs_data" | "no_model";
  datasetId?: string;
  okCount: number;
  faultCount: number;
  pieceCount: number;
  confidence: number | null;
  falsePassRate: number | null;
  createdAt?: string;
  lastActionAt?: string;
  lastTraining?: string;
  lastInspection?: string;
};

type LeadershipMetrics = {
  totalValidations: number;
  correctCount: number;
  reviewCount: number;
  retakeCount: number;
  retakeRate: number;
  activeMolds: number;
  missingPieces: number;
  latestTraining?: string;
  falsePassRate: number | null;
  validationRecall: number | null;
};

type MoldValidationSummary = {
  id: string;
  name: string;
  family: string;
  zoneId: string;
  status: MoldSummary["status"];
  validations: number;
  correct: number;
  review: number;
  retake: number;
  retakeRate: number;
  missingPieces: number;
  lastInspection?: string;
  lastTraining?: string;
  confidence: number | null;
  falsePassRate: number | null;
  validationRecall: number | null;
  latestImageUri?: string;
  latestMessage?: string;
};

type InspectionTrendPoint = {
  date: string;
  total: number;
  correct: number;
  review: number;
  retake: number;
};

type NormalizedInspection = {
  id: string;
  family: string;
  zoneId: string;
  status: "correct" | "review" | "retake_photo";
  createdAt?: string;
  message?: string;
  imageUri?: string;
  missingCount: number;
};


type PieceFinding = {
  status?: "present" | "missing" | "uncertain" | string;
  region?: Point[];
  polygon_normalized?: Point[];
  polygon?: Point[];
  bbox_normalized?: { x: number; y: number; width: number; height: number };
};

type InspectionResult = {
  id: string;
  status: "correct" | "review" | "retake_photo";
  message: string;
  guidance: string[];
  evidence: Record<string, string>;
  identified_mold?: string | null;
  identified_zone?: string | null;
  confidence?: number | null;
  mold_polygon?: Point[];
  missing_regions?: Point[][];
  overlay_image_uri?: string | null;
  result: {
    anomaly_score?: number | null;
    anomaly_threshold?: number | null;
    model_id?: string | null;
    model_version?: string | null;
    capture_quality?: Record<string, unknown>;
    difference_regions?: unknown[];
    mold_segmentation?: {
      polygon_normalized?: Point[];
      confidence?: number;
    };
    piece_inspection?: {
      status?: string;
      findings?: PieceFinding[];
      missing_count?: number;
      present_count?: number;
      uncertain_count?: number;
    };
  };
};

type CaptureGuidanceResult = {
  ok: boolean;
  auto_capture_ready: boolean;
  message: string;
  guidance: string[];
  quality: Record<string, unknown>;
  alignment: {
    mold_segmentation?: {
      polygon_normalized?: Point[];
      confidence?: number;
    };
    [key: string]: unknown;
  };
};

type MaskRect = {
  left: number;
  top: number;
  right: number;
  bottom: number;
};

type MoldDrafts = {
  names: Record<string, string>;
  deleted: string[];
};

type MoldViewSide = "left" | "right" | "front";

type MoldSection = {
  id: string;
  zoneId: string;
  label: string;
  zoneIndex: number;
  view: MoldViewSide;
};

type MoldSectionPlan = {
  moldKey: string;
  family: string;
  sections: MoldSection[];
  updatedAt: string;
};

type SectionResult = {
  status: "correct" | "review" | "retake_photo";
  message: string;
  updatedAt: string;
  inspectionId?: string | null;
  imageUri?: string | null;
};

type MoldValidationSession = {
  id: string;
  family: string;
  mold_key?: string;
  status: "pending" | "in_progress" | "complete";
  required_count: number;
  completed_count: number;
  missing_section_ids: string[];
  ready_section_ids: string[];
  section_results: Record<string, Record<string, unknown>>;
};

const SUPABASE_ENABLED = supabaseConfigured && import.meta.env.VITE_ENABLE_SUPABASE === "true" && !isLocalRuntime();
const DEFAULT_ZONES: Zone[] = [{ id: "frontal_zona_01", family: "molde_demo", name: "Frontal zona 01" }];
const VIEW_OPTIONS: Array<{ value: MoldViewSide; label: string }> = [
  { value: "left", label: "Izquierda" },
  { value: "right", label: "Derecha" },
  { value: "front", label: "Frente" }
];
const DEFAULT_SEGMENTER_POLYGON: Point[] = [
  { x: 0.18, y: 0.22 },
  { x: 0.82, y: 0.18 },
  { x: 0.88, y: 0.72 },
  { x: 0.28, y: 0.82 },
  { x: 0.14, y: 0.56 }
];

function App() {
  const [view, setView] = useState<View>("capture");
  const [family, setFamily] = useState("molde_demo");
  const [zoneId, setZoneId] = useState("frontal_zona_01");
  const [moldId, setMoldId] = useState("");
  const [selectedCaptureMold, setSelectedCaptureMold] = useState<MoldSummary | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [captureOpen, setCaptureOpen] = useState(false);
  const [status, setStatus] = useState<InspectionStatus>("idle");
  const [message, setMessage] = useState("Lista para capturar una zona.");
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [zoneReference, setZoneReference] = useState<ZoneReference | null>(null);
  const [expectedPieces, setExpectedPieces] = useState<ExpectedPiece[]>([]);
  const [sectionPlan, setSectionPlan] = useState<MoldSectionPlan>(() => loadSectionPlan(family, moldId, zoneId));
  const [sectionPlanSync, setSectionPlanSync] = useState<"local" | "saving" | "saved" | "error">("local");
  const [activeSectionId, setActiveSectionId] = useState(sectionPlan.sections[0]?.id || "");
  const [sectionResults, setSectionResults] = useState<Record<string, SectionResult>>({});
  const [validationSession, setValidationSession] = useState<MoldValidationSession | null>(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);
  const activeSection = sectionPlan.sections.find((section) => section.id === activeSectionId) || sectionPlan.sections[0] || null;
  const activeSectionKey = activeSection?.id || zoneId;

  useEffect(() => {
    void loadCaptureContext(family, zoneId).then(({ reference, expected }) => {
      setZoneReference(reference);
      setExpectedPieces(expected);
    });
  }, [family, zoneId]);

  useEffect(() => {
    const nextPlan = loadSectionPlan(family, moldId, zoneId);
    setSectionPlan(nextPlan);
    const nextSection = nextPlan.sections.find((section) => section.zoneId === zoneId) || nextPlan.sections[0] || null;
    setActiveSectionId(nextSection?.id || "");
    if (nextSection && nextSection.zoneId !== zoneId) setZoneId(nextSection.zoneId);
    setSectionPlanSync("local");

    let cancelled = false;
    void loadRemoteSectionPlan(family, moldId || family, zoneId).then((remotePlan) => {
      if (cancelled || !remotePlan) return;
      persistSectionPlan(remotePlan);
      setSectionPlan(remotePlan);
      const remoteSection = remotePlan.sections.find((section) => section.zoneId === zoneId) || remotePlan.sections[0] || null;
      setActiveSectionId(remoteSection?.id || "");
      if (remoteSection && remoteSection.zoneId !== zoneId) setZoneId(remoteSection.zoneId);
      setSectionPlanSync("saved");
    }).catch(() => {
      if (!cancelled) setSectionPlanSync("local");
    });
    return () => { cancelled = true; };
  }, [family, moldId]);

  useEffect(() => {
    let cancelled = false;
    void ensureValidationSession(sectionPlan).then((session) => {
      if (cancelled) return;
      setValidationSession(session);
      if (session) setSectionResults(validationResultsFromSession(session));
    }).catch(() => {
      if (!cancelled) setValidationSession(null);
    });
    return () => { cancelled = true; };
  }, [sectionPlan.family, sectionPlan.moldKey, sectionPlan.updatedAt]);

  async function runInspection(targetFile = file) {
    if (!targetFile) {
      setStatus("error");
      setMessage("Selecciona o toma una foto antes de validar.");
      return;
    }
    const clientQuality = await validateClientImage(targetFile);
    if (!clientQuality.ok) {
      setResult(null);
      setStatus("retake_photo");
      setMessage(clientQuality.message);
      setSectionResults((current) => ({ ...current, [activeSectionKey]: { status: "retake_photo", message: clientQuality.message, updatedAt: new Date().toISOString() } }));
      return;
    }
    setResult(null);
    setStatus("uploading");
    setMessage("Subiendo imagen...");
    try {
      const presign = await postJson("/v1/uploads/presign", {
        filename: targetFile.name,
        content_type: targetFile.type || "image/jpeg",
        family,
        zone_id: zoneId,
        purpose: "inspection"
      });
      await fetch(resolveUrl(presign.upload_url), {
        method: presign.method,
        headers: presign.headers,
        body: targetFile
      });
      setStatus("running");
      setMessage("Validando encuadre...");
      const alignment = (await postJson("/v1/uploads/align-quality", {
        family,
        zone_id: zoneId,
        image_uri: presign.object_uri,
        reference_id: zoneReference?.reference_id || null
      })) as { status: InspectionStatus; ok: boolean; message: string; guidance?: string[] };
      if (!alignment.ok) {
        setStatus("retake_photo");
        setMessage(alignment.message);
        setSectionResults((current) => ({ ...current, [activeSectionKey]: { status: "retake_photo", message: alignment.message, updatedAt: new Date().toISOString() } }));
        const nextSession = await saveValidationProgress({
          status: "retake_photo",
          image_uri: presign.object_uri,
          message: alignment.message
        });
        return;
      }
      setMessage("Validando captura y modelo...");
      const inspection = (await postJson("/v1/inspections", {
        family,
        zone_id: zoneId,
        mold_id: moldId || null,
        session_id: null,
        image_uri: presign.object_uri,
        capture_metadata: {
          source: "web",
          reference_id: zoneReference?.reference_id || null,
          client_quality: clientQuality
        }
      })) as InspectionResult;
      setResult(inspection);
      setStatus(inspection.status);
      setMessage(inspection.message);
      setSectionResults((current) => ({ ...current, [activeSectionKey]: { status: inspection.status, message: inspection.message, updatedAt: new Date().toISOString() } }));
      const nextSession = await saveValidationProgress({
        status: inspection.status,
        inspection_id: inspection.id,
        image_uri: presign.object_uri,
        message: inspection.message
      });
      await insertSupabase("inspections", { ...toPlainRecord(inspection), image_uri: presign.object_uri, family, zone_id: zoneId, mold_id: moldId || null });
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Error inesperado.");
    }
  }

  async function handleCaptured(capturedFile: File) {
    setFile(capturedFile);
    setCaptureOpen(false);
    await runInspection(capturedFile);
  }

  function selectMold(mold: MoldSummary) {
    setSelectedCaptureMold(mold);
    setFamily(mold.family);
    setZoneId(mold.zoneId);
    setMoldId(mold.id);
    setFile(null);
    setResult(null);
    setStatus("idle");
    setMessage("Lista para capturar una zona.");
  }

  function startSectionCapture(section: MoldSection) {
    activateSection(section);
    setCaptureOpen(true);
  }

  function activateSection(section: MoldSection) {
    setActiveSectionId(section.id);
    setZoneId(section.zoneId);
    setFile(null);
    setResult(null);
    setStatus("idle");
    setMessage("Lista para capturar una zona.");
  }

  function saveSectionPlan(nextPlan: MoldSectionPlan) {
    persistSectionPlan(nextPlan);
    setSectionPlan(nextPlan);
    setSectionPlanSync("saving");
    const nextSection = nextPlan.sections[0] || null;
    if (nextSection) activateSection(nextSection);
    void persistRemoteSectionPlan(nextPlan).then((remotePlan) => {
      if (remotePlan) {
        persistSectionPlan(remotePlan);
        setSectionPlan(remotePlan);
      }
      setSectionPlanSync("saved");
    }).catch(() => setSectionPlanSync("error"));
  }

  async function saveValidationProgress(payload: {
    status: "correct" | "review" | "retake_photo";
    inspection_id?: string;
    image_uri?: string;
    message?: string;
  }) {
    const sessionForWrite = validationSession || await ensureValidationSession(sectionPlan);
    if (!sessionForWrite) return null;
    const nextSession = await recordValidationSectionResult(sessionForWrite, activeSection, payload);
    if (nextSession) {
      setValidationSession(nextSession);
      setSectionResults(validationResultsFromSession(nextSession));
      if (nextSession.status === "complete") setMessage("Molde completo: todas las vistas requeridas tienen foto aceptada o revisada.");
    }
    return nextSession;
  }

  return (
    <main className="shell">
      <header className="appTopNav">
        <div className="brand">
          <span className="mark">3.2</span>
          <div>
            <strong>Inspeccion de moldes</strong>
            <small>Beta</small>
          </div>
        </div>
        <nav aria-label="Navegación principal">
          <NavButton icon={Camera} label="Captura" active={view === "capture"} onClick={() => setView("capture")} />
          <NavButton icon={PencilRuler} label="Moldes" active={view === "molds"} onClick={() => setView("molds")} />
          <NavButton icon={LayoutDashboard} label="Validaciones" active={view === "validations"} onClick={() => setView("validations")} />
        </nav>
        <details className="userMenu">
          <summary aria-label="Menú de usuario">
            <UserCircle size={18} />
            <span>Usuario</span>
          </summary>
          <div className="userMenuPanel">
            <strong>Operador</strong>
            <small>moldvision.local</small>
            <button type="button"><Settings size={14} /> Ajustes</button>
            <button type="button"><LogOut size={14} /> Salir</button>
          </div>
        </details>
      </header>

      <section className="workspace">
        {view === "molds" && <MoldsView onTest={(mold) => { selectMold(mold); setView("capture"); }} />}

        {view === "capture" && (
          <>
            <header className="topbar captureTopbar">
              <div>
                <h1>Captura</h1>
                <p>Busca molde. Elige zona. Captura.</p>
              </div>
            </header>

            <CaptureMoldSearch selectedMold={selectedCaptureMold} onSelect={selectMold} fallbackFamily={family} fallbackZoneId={zoneId} />

            {selectedCaptureMold ? (
              <MoldCaptureZones
                mold={selectedCaptureMold}
                sections={sectionPlan.sections}
                activeSectionId={activeSectionId}
                sectionResults={sectionResults}
                validationSession={validationSession}
                disabled={status === "uploading" || status === "running"}
                onSelectSection={activateSection}
                onCapture={startSectionCapture}
              />
            ) : null}

            {result ? <InspectionResultPanel status={status} message={message} result={result} previewUrl={previewUrl} family={family} zoneId={zoneId} /> : null}
            {captureOpen ? <CameraCapture family={family} zoneId={zoneId} previewUrl={previewUrl} reference={zoneReference} expectedPieces={expectedPieces} onFile={setFile} onCapture={handleCaptured} onCancel={() => setCaptureOpen(false)} fullscreen /> : null}
          </>
        )}

        {view === "validations" && <LeadershipDashboard />}
      </section>
    </main>
  );
}

function NavButton({ icon: Icon, label, active, onClick }: { icon: React.ElementType; label: string; active: boolean; onClick: () => void }) {
  return (
    <button className={active ? "active" : ""} onClick={onClick} title={label}>
      <Icon size={16} strokeWidth={1.8} />
      <span>{label}</span>
    </button>
  );
}

function SectionWorkflowPanel({
  family,
  zoneId,
  reference,
  expectedPieces,
  result,
  status,
  sections,
  activeSectionId,
  sectionResults,
  validationSession,
  onSelectSection
}: {
  family: string;
  zoneId: string;
  reference: ZoneReference | null;
  expectedPieces: ExpectedPiece[];
  result: InspectionResult | null;
  status: InspectionStatus;
  sections: MoldSection[];
  activeSectionId: string;
  sectionResults: Record<string, SectionResult>;
  validationSession: MoldValidationSession | null;
  onSelectSection: (section: MoldSection) => void;
}) {
  const pieces = expectedPieces.length ? expectedPieces : defaultExpectedPieces();
  const completedSections = validationSession?.completed_count ?? sections.filter((section) => sectionResults[section.id]?.status === "correct" || sectionResults[section.id]?.status === "review").length;
  const requiredSections = validationSession?.required_count ?? sections.length;
  const steps = [
    { label: "Golden sample", done: Boolean(reference), detail: reference?.reference_id || "pendiente" },
    { label: "Piezas esperadas", done: pieces.length > 0, detail: `${pieces.length} clases` },
    { label: "Captura", done: Boolean(result), detail: result ? statusLabel(result.status) : statusLabel(status) },
    { label: "Molde completo", done: validationSession?.status === "complete", detail: validationSession ? validationStatusLabel(validationSession.status) : "sin sesión" }
  ];
  return (
    <section className="sectionWorkflow panel">
      <div>
        <span>Sección actual</span>
        <strong>{family} / {zoneId}</strong>
        <b>{completedSections}/{requiredSections || 1} vistas listas</b>
        <small>{validationSession ? `Servidor: ${validationStatusLabel(validationSession.status)}` : "Servidor: pendiente"}</small>
      </div>
      <div className="sectionSteps" aria-label="Checklist de sección">
        {steps.map((step) => (
          <div key={step.label} className={step.done ? "done" : ""}>
            <i />
            <span>{step.label}</span>
            <b>{step.detail}</b>
          </div>
        ))}
      </div>
      <div className="sectionMatrix" aria-label="Vistas requeridas">
        {sections.map((section) => {
          const sectionResult = sectionResults[section.id];
          return (
            <button key={section.id} type="button" className={section.id === activeSectionId ? "active" : ""} onClick={() => onSelectSection(section)}>
              <i className={sectionResult?.status || ""} />
              <span>{section.label}</span>
              <b>{sectionResult ? statusLabel(sectionResult.status) : "pendiente"}</b>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function SectionPlanControl({
  plan,
  activeSectionId,
  syncState,
  onPlanChange,
  onSelectSection
}: {
  plan: MoldSectionPlan;
  activeSectionId: string;
  syncState: "local" | "saving" | "saved" | "error";
  onPlanChange: (plan: MoldSectionPlan) => void;
  onSelectSection: (section: MoldSection) => void;
}) {
  const [zoneCount, setZoneCount] = useState(() => Math.max(1, Math.max(...plan.sections.map((section) => section.zoneIndex), 1)));
  const [views, setViews] = useState<MoldViewSide[]>(() => selectedViewsFromPlan(plan));

  useEffect(() => {
    setZoneCount(Math.max(1, Math.max(...plan.sections.map((section) => section.zoneIndex), 1)));
    setViews(selectedViewsFromPlan(plan));
  }, [plan.moldKey, plan.updatedAt]);

  function toggleView(view: MoldViewSide) {
    setViews((current) => current.includes(view) ? current.filter((item) => item !== view) : [...current, view]);
  }

  function applyPlan() {
    const selectedViews = views.length ? views : ["front" as MoldViewSide];
    onPlanChange(buildSectionPlan(plan.family, plan.moldKey, zoneCount, selectedViews));
  }

  return (
    <section className="sectionPlan panel">
      <div>
        <span>Separación del molde</span>
        <strong>{plan.sections.length} vistas requeridas</strong>
        <small className={`syncState ${syncState}`}>{syncStateLabel(syncState)}</small>
      </div>
      <label>Zonas
        <input type="number" min={1} max={12} value={zoneCount} onChange={(event) => setZoneCount(clamp(Math.round(Number(event.target.value) || 1), 1, 12))} />
      </label>
      <div className="viewToggles" aria-label="Vistas por zona">
        {VIEW_OPTIONS.map((option) => (
          <label key={option.value}>
            <input type="checkbox" checked={views.includes(option.value)} onChange={() => toggleView(option.value)} />
            <span>{option.label}</span>
          </label>
        ))}
      </div>
      <button className="secondary" type="button" onClick={applyPlan}>Aplicar separación</button>
      <select value={activeSectionId} onChange={(event) => {
        const section = plan.sections.find((item) => item.id === event.target.value);
        if (section) onSelectSection(section);
      }}>
        {plan.sections.map((section) => <option key={section.id} value={section.id}>{section.label}</option>)}
      </select>
    </section>
  );
}

function MoldsView({ onTest }: { onTest: (mold: MoldSummary) => void }) {
  const [recipes, setRecipes] = useState<ResourceRecord[]>([]);
  const [datasets, setDatasets] = useState<ResourceRecord[]>([]);
  const [candidates, setCandidates] = useState<ResourceRecord[]>([]);
  const [jobs, setJobs] = useState<ResourceRecord[]>([]);
  const [inspections, setInspections] = useState<ResourceRecord[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [newMoldName, setNewMoldName] = useState("");
  const [query, setQuery] = useState("");
  const [sortBy, setSortBy] = useState<"recent" | "action" | "created" | "name">("recent");
  const [drafts, setDrafts] = useState<MoldDrafts>(() => loadMoldDrafts());
  const [editingName, setEditingName] = useState("");
  const [editing, setEditing] = useState(false);
  const [annotationMoldName, setAnnotationMoldName] = useState("");
  const [annotationOpen, setAnnotationOpen] = useState(false);
  const [detailOpen, setDetailOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const molds = useMemo(() => {
    return buildMoldSummaries(recipes, datasets, candidates, jobs, inspections)
      .filter((mold) => !drafts.deleted.includes(mold.id))
      .map((mold) => ({ ...mold, name: drafts.names[mold.id] || mold.name }));
  }, [recipes, datasets, candidates, jobs, inspections, drafts]);
  const visibleMolds = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const filtered = molds.filter((mold) => {
      if (!normalizedQuery) return true;
      return `${mold.name} ${mold.family} ${mold.zoneId} ${moldStatusLabel(mold.status)}`.toLowerCase().includes(normalizedQuery);
    });
    return [...filtered].sort((left, right) => {
      if (sortBy === "name") return left.name.localeCompare(right.name, "es");
      if (sortBy === "created") return dateRank(right.createdAt) - dateRank(left.createdAt);
      if (sortBy === "action") return dateRank(right.lastActionAt || right.lastInspection || right.lastTraining || right.createdAt) - dateRank(left.lastActionAt || left.lastInspection || left.lastTraining || left.createdAt);
      return dateRank(right.lastInspection || right.lastActionAt || right.createdAt) - dateRank(left.lastInspection || left.lastActionAt || left.createdAt);
    });
  }, [molds, query, sortBy]);
  const selectedMold = molds.find((mold) => mold.id === selectedId) || molds[0] || null;
  const selectedReadings = useMemo(() => {
    if (!selectedMold) return [];
    return inspectionSources(inspections)
      .filter((inspection) => inspection.family === selectedMold.family && inspection.zoneId === selectedMold.zoneId)
      .sort((left, right) => Date.parse(right.createdAt || "") - Date.parse(left.createdAt || ""));
  }, [inspections, selectedMold]);
  const selectedReadingStats = useMemo(() => readingStats(selectedReadings), [selectedReadings]);
  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (!selectedId && molds[0]) setSelectedId(molds[0].id);
  }, [molds, selectedId]);

  useEffect(() => {
    if (selectedMold && !editing) setEditingName(selectedMold.name);
  }, [selectedMold, editing]);

  useEffect(() => {
    document.body.classList.toggle("annotation-modal-open", annotationOpen);
    if (annotationOpen) window.scrollTo(0, 0);
    return () => document.body.classList.remove("annotation-modal-open");
  }, [annotationOpen]);

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const [loadedRecipes, loadedDatasets, loadedCandidates, loadedJobs, loadedInspections] = await Promise.all([
        loadRecords("recipes", "/v1/recipes"),
        loadRecords("datasets", "/v1/datasets"),
        loadRecords("model_candidates", "/v1/model_candidates"),
        loadRecords("inspector_training_jobs", "/v1/inspector_training_jobs"),
        loadRecords("inspections", "/v1/inspections")
      ]);
      setRecipes(loadedRecipes);
      setDatasets(loadedDatasets);
      setCandidates(loadedCandidates);
      setJobs(loadedJobs);
      setInspections(loadedInspections);
    } catch (loadError) {
      setError(messageFrom(loadError));
    } finally {
      setLoading(false);
    }
  }

  function openAnnotation(mold?: MoldSummary | null) {
    setAnnotationMoldName(mold?.name || newMoldName.trim() || "Nuevo molde");
    setAnnotationOpen(true);
  }

  async function updateMoldName(mold: MoldSummary) {
    const name = editingName.trim();
    if (!name) {
      setError("El nombre del molde no puede quedar vacío.");
      return;
    }
    const nextDrafts = { names: { ...drafts.names, [mold.id]: name }, deleted: drafts.deleted };
    saveMoldDrafts(nextDrafts);
    setDrafts(nextDrafts);
    setEditing(false);
    setError("");
    const recipe = recipes.find((item) => {
      const source = recordSource(item);
      return String(source.family || "") === mold.family && String(source.zone_id || DEFAULT_ZONES[0].id) === mold.zoneId;
    });
    if (SUPABASE_ENABLED && recipe?.id) {
      await supabase.from("recipes").update({ name }).eq("id", recipe.id);
    }
  }

  async function deleteMold(mold: MoldSummary) {
    if (!window.confirm(`¿Borrar "${mold.name}" de la lista de moldes?`)) return;
    const nextDrafts = {
      names: { ...drafts.names },
      deleted: Array.from(new Set([...drafts.deleted, mold.id]))
    };
    delete nextDrafts.names[mold.id];
    saveMoldDrafts(nextDrafts);
    setDrafts(nextDrafts);
    setSelectedId("");
    const recipe = recipes.find((item) => {
      const source = recordSource(item);
      return String(source.family || "") === mold.family && String(source.zone_id || DEFAULT_ZONES[0].id) === mold.zoneId;
    });
    if (SUPABASE_ENABLED && recipe?.id) {
      await supabase.from("recipes").delete().eq("id", recipe.id);
    }
  }

  return (
    <>
      <header className="topbar moldsTopbar">
        <div>
          <h1>Moldes</h1>
          <p>Gestiona moldes, revisa su estado y entra al flujo de anotación.</p>
        </div>
        <div className="moldsTopActions">
          <input value={newMoldName} onChange={(event) => setNewMoldName(event.target.value)} placeholder="Nombre nuevo molde" />
          <button className="primary" onClick={() => openAnnotation(null)} disabled={loading}>Nuevo molde</button>
          <button className="secondary" onClick={refresh} disabled={loading}>Actualizar</button>
        </div>
      </header>

      {error ? <p className="errorText inlineError">{error}</p> : null}

      <section className="moldsToolbar">
        <label className="moldSearchBox">
          <Search size={14} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Buscar molde, familia o zona" />
        </label>
        <label>Orden
          <select value={sortBy} onChange={(event) => setSortBy(event.target.value as "recent" | "action" | "created" | "name")}>
            <option value="recent">Más recientes</option>
            <option value="action">Última acción</option>
            <option value="created">Fecha de creación</option>
            <option value="name">Nombre</option>
          </select>
        </label>
        <span>{visibleMolds.length} moldes</span>
      </section>

      <section className="moldsBoard">
        <div className="moldsCardGrid">
          {visibleMolds.length ? visibleMolds.map((mold) => (
            <button
              key={mold.id}
              className={selectedMold?.id === mold.id ? "moldDocCard active" : "moldDocCard"}
              onClick={() => setSelectedId(mold.id)}
              onDoubleClick={() => { setSelectedId(mold.id); setDetailOpen(true); }}
              type="button"
            >
              <div className="moldDocPreview">
                <span className="moldDocMark">MV</span>
                <div className="moldPreviewLines">
                  <i /><i /><i /><i /><i />
                </div>
                <span className={`moldState ${mold.status}`}>{moldStatusLabel(mold.status)}</span>
              </div>
              <div className="moldDocFooter">
                <strong>{mold.name}</strong>
                <span>{mold.lastInspection ? `Última inspección ${formatDate(mold.lastInspection)}` : "Sin inspecciones"}</span>
              </div>
            </button>
          )) : (
            <div className="emptyMolds moldDocEmpty">
              <strong>{molds.length ? "Sin resultados" : "Sin moldes registrados"}</strong>
              <span>{molds.length ? "Ajusta la búsqueda o el orden." : "Crea el primer molde para configurar secciones y vistas."}</span>
              <button className="primary" type="button" onClick={() => openAnnotation(null)}>Nuevo molde</button>
            </div>
          )}
        </div>

        <aside className="moldInfoPanel">
          {selectedMold ? (
            <>
              <div className="moldInfoHeader">
                <span className={`status ${statusClassForMold(selectedMold.status)}`}>{moldStatusLabel(selectedMold.status)}</span>
                {editing ? (
                  <div className="editMoldName">
                    <input value={editingName} onChange={(event) => setEditingName(event.target.value)} aria-label="Nombre del molde" />
                    <button className="primary" onClick={() => updateMoldName(selectedMold)} disabled={loading}>Guardar</button>
                    <button className="secondary" onClick={() => setEditing(false)}>Cancelar</button>
                  </div>
                ) : <h2>{selectedMold.name}</h2>}
                <p>{selectedMold.lastInspection ? `Última lectura ${formatDate(selectedMold.lastInspection)}` : "Sin lecturas todavía."}</p>
              </div>

              <div className="moldInfoMetrics">
                <Metric label="Lecturas" value={String(selectedReadingStats.total)} />
                <Metric label="Error" value={formatPercent(selectedReadingStats.errorRate)} />
                <Metric label="Retoma" value={formatPercent(selectedReadingStats.retakeRate)} />
              </div>

              <div className="moldInfoActions">
                <button className="primary" type="button" onClick={() => openAnnotation(selectedMold)}>Anotar</button>
                <button className="secondary" onClick={() => onTest(selectedMold)}>Capturar foto</button>
                <button className="secondary" onClick={() => setDetailOpen(true)}>Ver datos</button>
                <button className="secondary" onClick={() => { setEditingName(selectedMold.name); setEditing(true); }}>Editar</button>
                <button className="dangerButton" onClick={() => deleteMold(selectedMold)}>Borrar</button>
              </div>

              <section className="moldInfoBlock moldReadingsBlock">
                <strong>Últimas lecturas</strong>
                {selectedReadings.length ? selectedReadings.slice(0, 5).map((reading) => (
                  <article key={reading.id} className="readingRow">
                    <span className={`dot ${reading.status}`} />
                    <div>
                      <b>{statusLabel(reading.status)}</b>
                      <small>{reading.createdAt ? formatDate(reading.createdAt) : "Sin fecha"}</small>
                    </div>
                    <em>{reading.missingCount} falt.</em>
                  </article>
                )) : <span>Sin lecturas registradas.</span>}
              </section>
            </>
          ) : (
            <div className="emptyMolds">
              <strong>Selecciona un molde</strong>
              <span>La información aparecerá aquí.</span>
            </div>
          )}
        </aside>
      </section>

      {annotationOpen ? (
        <div className="annotationModalHost" role="dialog" aria-modal="true">
          <AnnotateApp initialMoldName={annotationMoldName} onExit={() => setAnnotationOpen(false)} />
        </div>
      ) : null}
      {detailOpen && selectedMold ? (
        <MoldDetailDrawer mold={selectedMold} readings={selectedReadings} stats={selectedReadingStats} onClose={() => setDetailOpen(false)} />
      ) : null}
    </>
  );
}

function MoldDetailDrawer({
  mold,
  readings,
  stats,
  onClose
}: {
  mold: MoldSummary;
  readings: NormalizedInspection[];
  stats: ReturnType<typeof readingStats>;
  onClose: () => void;
}) {
  return (
    <div className="moldDetailOverlay" role="dialog" aria-modal="true">
      <aside className="moldDetailDrawer">
        <header>
          <div>
            <span className={`status ${statusClassForMold(mold.status)}`}>{moldStatusLabel(mold.status)}</span>
            <h2>{mold.name}</h2>
            <p>{mold.family} / {mold.zoneId}</p>
          </div>
          <button className="secondary" type="button" onClick={onClose}>Cerrar</button>
        </header>

        <div className="moldDetailMetrics">
          <Metric label="Lecturas" value={String(stats.total)} />
          <Metric label="Correctas" value={String(stats.correct)} />
          <Metric label="Revisión" value={String(stats.review)} />
          <Metric label="Retomas" value={String(stats.retake)} />
          <Metric label="Error" value={formatPercent(stats.errorRate)} />
          <Metric label="Piezas faltantes" value={String(stats.missingPieces)} />
        </div>

        <section className="moldInfoBlock">
          <strong>Datos clave</strong>
          <div className="keyDataGrid">
            <span><b>Confianza</b>{formatPercent(mold.confidence)}</span>
            <span><b>False pass</b>{formatPercent(mold.falsePassRate)}</span>
            <span><b>Último training</b>{mold.lastTraining ? formatDate(mold.lastTraining) : "-"}</span>
            <span><b>Última lectura</b>{mold.lastInspection ? formatDate(mold.lastInspection) : "-"}</span>
          </div>
        </section>

        <section className="moldInfoBlock">
          <strong>Histórico reciente</strong>
          <div className="readingHistory">
            {readings.length ? readings.slice(0, 12).map((reading) => (
              <article key={reading.id} className="readingRow detailed">
                <span className={`dot ${reading.status}`} />
                <div>
                  <b>{statusLabel(reading.status)}</b>
                  <small>{reading.message || "Lectura registrada"}</small>
                </div>
                <time>{reading.createdAt ? formatDate(reading.createdAt) : "-"}</time>
                <em>{reading.missingCount} falt.</em>
              </article>
            )) : <span className="emptyText compact">Sin lecturas para este molde.</span>}
          </div>
        </section>
      </aside>
    </div>
  );
}

function CaptureMoldSearch({
  selectedMold,
  onSelect,
  fallbackFamily,
  fallbackZoneId
}: {
  selectedMold: MoldSummary | null;
  onSelect: (mold: MoldSummary) => void;
  fallbackFamily: string;
  fallbackZoneId: string;
}) {
  const [molds, setMolds] = useState<MoldSummary[]>([]);
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    async function load() {
      const [recipes, datasets, candidates, jobs, inspections] = await Promise.all([
        loadRecords("recipes", "/v1/recipes"),
        loadRecords("datasets", "/v1/datasets"),
        loadRecords("model_candidates", "/v1/model_candidates"),
        loadRecords("inspector_training_jobs", "/v1/inspector_training_jobs"),
        loadRecords("inspections", "/v1/inspections")
      ]);
      setMolds(buildMoldSummaries(recipes, datasets, candidates, jobs, inspections));
    }
    void load();
  }, []);

  useEffect(() => {
    if (selectedMold) setQuery(selectedMold.name);
  }, [selectedMold]);

  const visibleMolds = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    const matched = normalized
      ? molds.filter((mold) => `${mold.name} ${mold.family} ${mold.zoneId}`.toLowerCase().includes(normalized))
      : molds;
    return matched.slice(0, 8);
  }, [molds, query]);

  function selectFirstMatch() {
    const first = visibleMolds[0] || molds[0] || {
      id: `${fallbackFamily}:${fallbackZoneId}`,
      name: titleFromSlug(fallbackFamily),
      family: fallbackFamily,
      zoneId: fallbackZoneId,
      status: "needs_data" as MoldSummary["status"],
      okCount: 0,
      faultCount: 0,
      pieceCount: 0,
      confidence: null,
      falsePassRate: null
    };
    onSelect(first);
    setFocused(false);
  }

  return (
    <section className="captureSearchOnly">
      <div className="captureSearchBox">
        <Search size={18} />
        <input
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setFocused(true);
          }}
          onFocus={() => setFocused(true)}
          onKeyDown={(event) => {
            if (event.key === "Enter") selectFirstMatch();
            if (event.key === "Escape") setFocused(false);
          }}
          placeholder="Buscar molde"
          aria-label="Buscar molde"
        />
      </div>
      {focused ? (
        <div className="captureSearchResults">
          {visibleMolds.length ? visibleMolds.map((mold) => (
            <button key={mold.id} type="button" onMouseDown={(event) => event.preventDefault()} onClick={() => { onSelect(mold); setFocused(false); }}>
              <strong>{mold.name}</strong>
              <span>{mold.family} / {mold.zoneId}</span>
            </button>
          )) : (
            <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={selectFirstMatch}>
              <strong>{titleFromSlug(fallbackFamily)}</strong>
              <span>{fallbackFamily} / {fallbackZoneId}</span>
            </button>
          )}
        </div>
      ) : null}
    </section>
  );
}

function MoldCaptureZones({
  mold,
  sections,
  activeSectionId,
  sectionResults,
  validationSession,
  disabled,
  onSelectSection,
  onCapture
}: {
  mold: MoldSummary;
  sections: MoldSection[];
  activeSectionId: string;
  sectionResults: Record<string, SectionResult>;
  validationSession: MoldValidationSession | null;
  disabled: boolean;
  onSelectSection: (section: MoldSection) => void;
  onCapture: (section: MoldSection) => void;
}) {
  const [pieceCounts, setPieceCounts] = useState<Record<string, number>>({});

  useEffect(() => {
    let cancelled = false;
    async function loadCounts() {
      const entries = await Promise.all(sections.map(async (section) => {
        const context = await loadCaptureContext(mold.family, section.zoneId).catch(() => ({ expected: [] as ExpectedPiece[] }));
        const count = context.expected.length || defaultExpectedPieces().length;
        return [section.id, count] as const;
      }));
      if (!cancelled) setPieceCounts(Object.fromEntries(entries));
    }
    void loadCounts();
    return () => { cancelled = true; };
  }, [mold.family, mold.id, sections]);

  const completedSections = validationSession?.completed_count ?? sections.filter((section) => {
    const state = sectionResults[section.id]?.status;
    return state === "correct" || state === "review";
  }).length;

  return (
    <section className="captureZones">
      <div className="captureZonesHeader">
        <div>
          <span>Molde</span>
          <strong>{mold.name}</strong>
          <small>{completedSections}/{sections.length || 1} zonas listas</small>
        </div>
        <span className={`status ${statusClassForMold(mold.status)}`}>{moldStatusLabel(mold.status)}</span>
      </div>
      <div className="captureZoneGrid">
        {sections.map((section) => {
          const result = sectionResults[section.id];
          const isActive = section.id === activeSectionId;
          return (
            <article key={section.id} className={isActive ? "captureZoneCard active" : "captureZoneCard"}>
              <button type="button" className="captureZoneBody" onClick={() => onSelectSection(section)}>
                <span>{section.label}</span>
                <strong>{(pieceCounts[section.id] ?? mold.pieceCount) || defaultExpectedPieces().length} piezas</strong>
                <small>{result ? statusLabel(result.status) : "pendiente"}</small>
              </button>
              <button className="primary" type="button" disabled={disabled} onClick={() => onCapture(section)}>
                Capturar
              </button>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function MoldQuickSelect({ family, zoneId, onSelect }: { family: string; zoneId: string; onSelect: (mold: MoldSummary) => void }) {
  return <CaptureMoldSearch selectedMold={null} onSelect={onSelect} fallbackFamily={family} fallbackZoneId={zoneId} />;
}

function LeadershipDashboard() {
  const [recipes, setRecipes] = useState<ResourceRecord[]>([]);
  const [datasets, setDatasets] = useState<ResourceRecord[]>([]);
  const [candidates, setCandidates] = useState<ResourceRecord[]>([]);
  const [jobs, setJobs] = useState<ResourceRecord[]>([]);
  const [inspections, setInspections] = useState<ResourceRecord[]>([]);
  const [statusFilter, setStatusFilter] = useState<"all" | "correct" | "review" | "retake_photo">("all");
  const [dateRange, setDateRange] = useState<"7" | "30" | "all">("30");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const moldSummaries = useMemo(
    () => buildMoldSummaries(recipes, datasets, candidates, jobs, inspections),
    [recipes, datasets, candidates, jobs, inspections]
  );
  const metrics = useMemo(() => buildLeadershipMetrics(moldSummaries, candidates, jobs, inspections), [moldSummaries, candidates, jobs, inspections]);
  const moldRows = useMemo(() => buildMoldValidationSummaries(moldSummaries, candidates, inspections), [moldSummaries, candidates, inspections]);
  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return moldRows.filter((row) => {
      const matchesQuery = !normalizedQuery || `${row.name} ${row.family} ${row.zoneId}`.toLowerCase().includes(normalizedQuery);
      const matchesStatus = statusFilter === "all" || (statusFilter === "correct" ? row.correct > 0 : statusFilter === "review" ? row.review > 0 : row.retake > 0);
      const matchesDate = dateRange === "all" || withinDays(row.lastInspection, Number(dateRange));
      return matchesQuery && matchesStatus && matchesDate;
    });
  }, [moldRows, query, statusFilter, dateRange]);
  const selectedRow = filteredRows.find((row) => row.id === selectedId) || filteredRows[0] || null;
  const selectedInspections = useMemo(() => {
    if (!selectedRow) return [];
    return inspectionSources(inspections)
      .filter((inspection) => inspection.family === selectedRow.family && inspection.zoneId === selectedRow.zoneId)
      .sort((left, right) => Date.parse(right.createdAt || "") - Date.parse(left.createdAt || ""))
      .slice(0, 6);
  }, [inspections, selectedRow]);
  const trend = useMemo(() => buildInspectionTrend(inspections), [inspections]);

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (!selectedId && filteredRows[0]) setSelectedId(filteredRows[0].id);
  }, [filteredRows, selectedId]);

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const [loadedRecipes, loadedDatasets, loadedCandidates, loadedJobs, loadedInspections] = await Promise.all([
        loadRecords("recipes", "/v1/recipes"),
        loadRecords("datasets", "/v1/datasets"),
        loadRecords("model_candidates", "/v1/model_candidates"),
        loadRecords("inspector_training_jobs", "/v1/inspector_training_jobs"),
        loadRecords("inspections", "/v1/inspections")
      ]);
      setRecipes(loadedRecipes);
      setDatasets(loadedDatasets);
      setCandidates(loadedCandidates);
      setJobs(loadedJobs);
      setInspections(loadedInspections);
    } catch (loadError) {
      setError(messageFrom(loadError));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <header className="topbar compactTopbar">
        <div>
          <span className="eyebrow">Dirección</span>
          <h1>Validaciones</h1>
          <p>Resultados por molde, uso de la app, retomas y salud de modelos.</p>
        </div>
        <button className="secondary iconButton" onClick={refresh} disabled={loading}><RefreshCw size={15} />Actualizar</button>
      </header>

      {error ? <p className="errorText inlineError">{error}</p> : null}

      <section className="dashboardMetrics">
        <KpiTile icon={ClipboardCheck} label="Validaciones" value={String(metrics.totalValidations)} detail={`${metrics.correctCount} correctas`} />
        <KpiTile icon={TriangleAlert} label="Revisión" value={String(metrics.reviewCount)} detail={`${metrics.missingPieces} piezas faltantes`} tone="warn" />
        <KpiTile icon={Camera} label="Retoma" value={formatPercent(metrics.retakeRate)} detail={`${metrics.retakeCount} fotos`} tone={metrics.retakeRate > 0.15 ? "danger" : "neutral"} />
        <KpiTile icon={Database} label="Moldes activos" value={String(metrics.activeMolds)} detail="family + zona" />
        <KpiTile icon={ShieldCheck} label="False pass" value={formatPercent(metrics.falsePassRate)} detail="modelo promovido" />
        <KpiTile icon={TrendingUp} label="Recall" value={formatPercent(metrics.validationRecall)} detail={metrics.latestTraining ? formatDate(metrics.latestTraining) : "sin training"} />
      </section>

      <section className="dashboardGrid">
        <div className="panel dashboardMain">
          <div className="tableToolbar">
            <div className="searchBox">
              <Search size={14} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Buscar molde, familia o zona" />
            </div>
            <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as "all" | "correct" | "review" | "retake_photo")}>
              <option value="all">Todos estados</option>
              <option value="correct">Correctos</option>
              <option value="review">Revisión</option>
              <option value="retake_photo">Retoma</option>
            </select>
            <select value={dateRange} onChange={(event) => setDateRange(event.target.value as "7" | "30" | "all")}>
              <option value="7">7 días</option>
              <option value="30">30 días</option>
              <option value="all">Todo</option>
            </select>
          </div>
          <div className="validationTable" role="table" aria-label="Validaciones por molde">
            <div className="validationTableHead" role="row">
              <span>Molde</span>
              <span>Valid.</span>
              <span>Correcto</span>
              <span>Revisión</span>
              <span>Retoma</span>
              <span>Modelo</span>
              <span>Última</span>
            </div>
            {filteredRows.length ? filteredRows.map((row) => (
              <button key={row.id} className={selectedRow?.id === row.id ? "validationRow active" : "validationRow"} onClick={() => setSelectedId(row.id)} role="row">
                <span><strong>{row.name}</strong><small>{row.family} / {row.zoneId}</small></span>
                <b>{row.validations}</b>
                <span>{row.correct}</span>
                <span>{row.review}</span>
                <span>{row.retake}</span>
                <span className={`status ${statusClassForMold(row.status)}`}>{moldStatusLabel(row.status)}</span>
                <time>{row.lastInspection ? formatDate(row.lastInspection) : "-"}</time>
              </button>
            )) : (
              <div className="dashboardEmpty">
                <CircleAlert size={20} />
                <strong>Sin datos para este filtro</strong>
                <span>Cuando haya inspecciones, dirección verá aquí validaciones por molde.</span>
              </div>
            )}
          </div>
        </div>

        <aside className="panel dashboardDetail">
          {selectedRow ? (
            <>
              <div className="detailHeader">
                <span className={`status ${statusClassForMold(selectedRow.status)}`}>{moldStatusLabel(selectedRow.status)}</span>
                <h2>{selectedRow.name}</h2>
                <p>{selectedRow.family} / {selectedRow.zoneId}</p>
              </div>
              <div className="detailStats">
                <Metric label="Validaciones" value={String(selectedRow.validations)} />
                <Metric label="Retoma" value={formatPercent(selectedRow.retakeRate)} />
                <Metric label="False pass" value={formatPercent(selectedRow.falsePassRate)} />
                <Metric label="Recall" value={formatPercent(selectedRow.validationRecall)} />
              </div>
              <div className="resultMix">
                <StatusBar label="Correcto" value={selectedRow.correct} total={Math.max(1, selectedRow.validations)} className="ok" />
                <StatusBar label="Revisión" value={selectedRow.review} total={Math.max(1, selectedRow.validations)} className="warn" />
                <StatusBar label="Retoma" value={selectedRow.retake} total={Math.max(1, selectedRow.validations)} className="bad" />
              </div>
              <div className="latestInspections">
                <strong>Últimas validaciones</strong>
                {selectedInspections.length ? selectedInspections.map((inspection) => (
                  <article key={inspection.id}>
                    <span className={`dot ${inspection.status}`} />
                    <div>
                      <b>{statusLabel(inspection.status)}</b>
                      <small>{inspection.message || "Inspección registrada"}</small>
                    </div>
                    <time>{inspection.createdAt ? formatDate(inspection.createdAt) : "-"}</time>
                  </article>
                )) : <p className="emptyText compact">Sin validaciones recientes.</p>}
              </div>
            </>
          ) : (
            <div className="dashboardEmpty">
              <Gauge size={22} />
              <strong>Sin molde seleccionado</strong>
              <span>Selecciona un molde para ver detalle.</span>
            </div>
          )}
        </aside>
      </section>

      <section className="panel trendPanel">
        <div className="captureHeader">
          <strong>Uso de app por día</strong>
          <span>{trend.reduce((total, item) => total + item.total, 0)} validaciones</span>
        </div>
        <div className="trendBars">
          {trend.map((point) => <TrendColumn key={point.date} point={point} max={Math.max(1, ...trend.map((item) => item.total))} />)}
        </div>
      </section>
    </>
  );
}

function KpiTile({ icon: Icon, label, value, detail, tone = "neutral" }: { icon: React.ElementType; label: string; value: string; detail: string; tone?: "neutral" | "warn" | "danger" }) {
  return (
    <article className={`kpiTile ${tone}`}>
      <Icon size={16} strokeWidth={1.8} />
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function StatusBar({ label, value, total, className }: { label: string; value: number; total: number; className: string }) {
  return (
    <div>
      <span>{label}</span>
      <i><b className={className} style={{ width: `${Math.round((value / total) * 100)}%` }} /></i>
      <strong>{value}</strong>
    </div>
  );
}

function TrendColumn({ point, max }: { point: InspectionTrendPoint; max: number }) {
  return (
    <div className="trendColumn">
      <i style={{ height: `${Math.max(6, Math.round((point.total / max) * 100))}%` }}>
        <b className="ok" style={{ height: `${point.total ? Math.round((point.correct / point.total) * 100) : 0}%` }} />
        <b className="warn" style={{ height: `${point.total ? Math.round((point.review / point.total) * 100) : 0}%` }} />
        <b className="bad" style={{ height: `${point.total ? Math.round((point.retake / point.total) * 100) : 0}%` }} />
      </i>
      <span>{shortDay(point.date)}</span>
    </div>
  );
}

function RecipesView({ family, zoneId, families, zones }: { family: string; zoneId: string; families: string[]; zones: Zone[] }) {
  const [selectedFamily, setSelectedFamily] = useState(family);
  const [selectedZone, setSelectedZone] = useState(zoneId);
  const [name, setName] = useState("Receta de inspección de molde");
  const [records, setRecords] = useState<ResourceRecord[]>([]);
  const [datasets, setDatasets] = useState<ResourceRecord[]>([]);
  const [candidates, setCandidates] = useState<ResourceRecord[]>([]);
  const [jobs, setJobs] = useState<ResourceRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const latestDataset = useMemo(() => {
    return datasets.find((item) => recordString(item, "family") === selectedFamily && recordString(item, "zone_id") === selectedZone);
  }, [datasets, selectedFamily, selectedZone]);
  const visibleCandidates = useMemo(() => {
    return candidates.filter((item) => recordString(item, "family") === selectedFamily && recordString(item, "zone_id") === selectedZone);
  }, [candidates, selectedFamily, selectedZone]);

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const [loadedRecipes, loadedDatasets, loadedCandidates, loadedJobs] = await Promise.all([
        loadRecords("recipes", "/v1/recipes"),
        loadRecords("datasets", "/v1/datasets"),
        loadRecords("model_candidates", "/v1/model_candidates"),
        loadRecords("inspector_training_jobs", "/v1/inspector_training_jobs")
      ]);
      setRecords(loadedRecipes);
      setDatasets(loadedDatasets);
      setCandidates(loadedCandidates);
      setJobs(loadedJobs);
    } catch (loadError) {
      setError(messageFrom(loadError));
    } finally {
      setLoading(false);
    }
  }

  async function createRecipe() {
    setLoading(true);
    setError("");
    try {
      const recipe = await postJson("/v1/recipes", {
        family: selectedFamily,
        zone_id: selectedZone,
        name,
        objective: "presence_absence"
      });
      await insertSupabase("recipes", recipe as Record<string, unknown>);
      await refresh();
    } catch (saveError) {
      setError(messageFrom(saveError));
      setLoading(false);
    }
  }

  async function startInspectorTraining() {
    if (!latestDataset) {
      setError("Primero registra un dataset ok/fault para esta familia/zona.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const job = await postJson("/v1/inspector-training-jobs", {
        family: selectedFamily,
        zone_id: selectedZone,
        dataset_id: String(latestDataset.id),
        target: "cloud-gpu"
      });
      await insertSupabase("inspector_training_jobs", job as Record<string, unknown>);
      await refresh();
    } catch (trainingError) {
      setError(messageFrom(trainingError));
      setLoading(false);
    }
  }

  async function promote(candidateId: string) {
    setLoading(true);
    setError("");
    try {
      await postJson(`/v1/model-candidates/${candidateId}/promote`, { notes: "Promovido desde AI Recipe." });
      await refresh();
    } catch (promoteError) {
      setError(messageFrom(promoteError));
      setLoading(false);
    }
  }

  return (
    <>
      <header className="topbar">
        <div>
          <h1>AI Recipe</h1>
          <p>Flujo tipo SolVision: datos, entrenamiento, métricas y modelo promovido para inspección.</p>
        </div>
        <button className="primary" onClick={refresh} disabled={loading}>Actualizar</button>
      </header>
      <section className="recipeGrid">
        <aside className="panel formPanel">
          <label>Nombre<input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <SelectField label="Familia" value={selectedFamily} onChange={setSelectedFamily} options={families.length ? families : [family]} />
          <SelectField label="Zona" value={selectedZone} onChange={setSelectedZone} options={zones.map((zone) => zone.id)} labels={Object.fromEntries(zones.map((zone) => [zone.id, zone.name || zone.label || zone.id]))} />
          <button className="primary fullWidth" onClick={createRecipe} disabled={loading}>Crear receta</button>
          <button className="secondary fullWidth" onClick={startInspectorTraining} disabled={loading}>Start Training</button>
          {error ? <p className="errorText compactError">{error}</p> : null}
        </aside>

        <section className="panel recipePanel">
          <div className="recipeSteps">
            <RecipeStep title="Training Data" status={latestDataset ? "ready" : "missing"} value={latestDataset ? `${recordString(latestDataset, "ok_count")} ok / ${recordString(latestDataset, "fault_count")} fault` : "sin dataset"} />
            <RecipeStep title="Mold Segmenter" status="ready" value="polígono + guía móvil" />
            <RecipeStep title="Inspector" status={visibleCandidates.some((item) => recordSource(item).promoted) ? "ready" : jobs.length ? "training" : "missing"} value={visibleCandidates.length ? `${visibleCandidates.length} candidatos` : "sin candidatos"} />
            <RecipeStep title="Start Inspection" status={visibleCandidates.some((item) => recordSource(item).promoted) ? "ready" : "missing"} value={visibleCandidates.some((item) => recordSource(item).promoted) ? "modelo promovido" : "esperando modelo"} />
          </div>
          <div className="candidateGrid">
            {visibleCandidates.length ? visibleCandidates.map((candidate) => {
              const source = recordSource(candidate);
              const metrics = (source.metrics || {}) as Record<string, unknown>;
              return (
                <article className={source.promoted ? "candidateCard promoted" : "candidateCard"} key={String(source.id)}>
                  <div className="captureHeader">
                    <strong>{String(source.name || source.id)}</strong>
                    <span className={`status ${source.promoted ? "correct" : "running"}`}>{source.promoted ? "best" : "candidate"}</span>
                  </div>
                  <div className="metrics compactMetrics">
                    <Metric label="Loss" value={formatUnknown(metrics.loss)} />
                    <Metric label="Confidence" value={formatPercentValue(metrics.confidence)} />
                    <Metric label="Recall" value={formatPercentValue(metrics.validation_recall)} />
                    <Metric label="False pass" value={formatPercentValue(metrics.false_pass_rate)} />
                  </div>
                  {!source.promoted ? <button className="secondary fullWidth" onClick={() => promote(String(source.id))} disabled={loading}>Promover</button> : null}
                </article>
              );
            }) : <p className="emptyText">Entrena el inspector para ver loss, confidence, recall y false-pass rate.</p>}
          </div>
        </section>

        <RecordPanel loading={loading} error="" records={records} hiddenKeys={["manifest_uri", "mask_uri", "model_uri"]} />
      </section>
    </>
  );
}

function RecipeStep({ title, status, value }: { title: string; status: "ready" | "training" | "missing"; value: string }) {
  return (
    <div className={`recipeStep ${status}`}>
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SegmenterView() {
  const [name, setName] = useState("Segmentador de moldes");
  const [files, setFiles] = useState<File[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [maskRect, setMaskRect] = useState<MaskRect>({ left: 22, top: 18, right: 78, bottom: 82 });
  const [records, setRecords] = useState<ResourceRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const selectedFile = files[selectedIndex] || files[0] || null;
  const previewUrl = useMemo(() => (selectedFile ? URL.createObjectURL(selectedFile) : ""), [selectedFile]);

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      setRecords(await loadRecords("segmenter_datasets", "/v1/segmenter_datasets"));
    } catch (loadError) {
      setError(messageFrom(loadError));
    } finally {
      setLoading(false);
    }
  }

  async function createDataset() {
    if (!files.length) {
      setError("Sube al menos una foto anotada del molde.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const imageUris = await uploadFiles(files, "generic_mold", "segmenter", "segmenter");
      const dataset = await postJson("/v1/segmenter-datasets/from-annotations", {
        name,
        annotations: imageUris.map((imageUri, index) => ({
          image_uri: imageUri,
          split: index === 1 ? "val" : "train",
          polygon: maskPoints(maskRect)
        }))
      }) as ResourceRecord;
      await insertSupabase("segmenter_datasets", dataset as Record<string, unknown>);
      await refresh();
    } catch (saveError) {
      setError(messageFrom(saveError));
      setLoading(false);
    }
  }

  async function trainLatest() {
    const dataset = records[0];
    if (!dataset) {
      setError("Primero registra un dataset del segmentador.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const job = await postJson("/v1/segmenter-training-jobs", { dataset_id: String(dataset.id), epochs: 50, image_size: 640 });
      await insertSupabase("segmenter_training_jobs", job as Record<string, unknown>);
      await refresh();
    } catch (trainError) {
      setError(messageFrom(trainError));
      setLoading(false);
    }
  }

  return (
    <>
      <header className="topbar">
        <div>
          <h1>Segmentador</h1>
          <p>Entrena la red que detecta el polígono del molde contra el fondo.</p>
        </div>
        <button className="primary" onClick={refresh} disabled={loading}>Actualizar</button>
      </header>
      <section className="datasetGrid">
        <aside className="panel formPanel">
          <label>Nombre<input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <label>Fotos anotadas<input type="file" accept="image/*" multiple onChange={(event) => setFiles(Array.from(event.target.files || []))} /></label>
          {files.length ? (
            <label>Foto para ajustar
              <select value={selectedIndex} onChange={(event) => setSelectedIndex(Number(event.target.value))}>
                {files.map((file, index) => <option key={`${file.name}-${index}`} value={index}>{file.name}</option>)}
              </select>
            </label>
          ) : null}
          <p className="fieldNote">El polígono se guarda como anotación de segmentación para la clase molde.</p>
          <button className="primary fullWidth" onClick={createDataset} disabled={loading}>Guardar anotaciones</button>
          <button className="secondary fullWidth" onClick={trainLatest} disabled={loading}>Entrenar segmentador</button>
        </aside>
        <section className="panel maskPanel">
          <div className="captureHeader">
            <strong>Polígono del molde</strong>
            <span className="status running">manual</span>
          </div>
          <div className="maskPreview">
            {previewUrl ? <img src={previewUrl} alt="Molde anotado" /> : <span>Sube una foto y ajusta el área del molde</span>}
            <div
              className="maskOverlay"
              style={{
                left: `${maskRect.left}%`,
                top: `${maskRect.top}%`,
                width: `${maskRect.right - maskRect.left}%`,
                height: `${maskRect.bottom - maskRect.top}%`
              }}
            />
            <div className="frame"><i /><i /><i /><i /></div>
          </div>
          <div className="maskControls">
            <RangeField label="Izquierda" value={maskRect.left} min={0} max={Math.min(maskRect.right - 5, 90)} onChange={(left) => setMaskRect((current) => ({ ...current, left }))} />
            <RangeField label="Derecha" value={maskRect.right} min={Math.max(maskRect.left + 5, 10)} max={100} onChange={(right) => setMaskRect((current) => ({ ...current, right }))} />
            <RangeField label="Arriba" value={maskRect.top} min={0} max={Math.min(maskRect.bottom - 5, 90)} onChange={(top) => setMaskRect((current) => ({ ...current, top }))} />
            <RangeField label="Abajo" value={maskRect.bottom} min={Math.max(maskRect.top + 5, 10)} max={100} onChange={(bottom) => setMaskRect((current) => ({ ...current, bottom }))} />
          </div>
        </section>
        <RecordPanel loading={loading} error={error} records={records} hiddenKeys={["dataset_uri", "data_yaml_uri"]} />
      </section>
    </>
  );
}

function DatasetView({ family, zoneId, families, zones }: { family: string; zoneId: string; families: string[]; zones: Zone[] }) {
  const [selectedFamily, setSelectedFamily] = useState(family);
  const [selectedZone, setSelectedZone] = useState(zoneId);
  const [name, setName] = useState("Dataset de referencia");
  const [okFiles, setOkFiles] = useState<File[]>([]);
  const [faultFiles, setFaultFiles] = useState<File[]>([]);
  const [referenceIndex, setReferenceIndex] = useState(0);
  const [maskMode, setMaskMode] = useState<"auto" | "manual">("auto");
  const [maskRect, setMaskRect] = useState<MaskRect>({ left: 18, top: 16, right: 82, bottom: 84 });
  const [records, setRecords] = useState<ResourceRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const referenceFile = okFiles[referenceIndex] || okFiles[0] || null;
  const referenceUrl = useMemo(() => (referenceFile ? URL.createObjectURL(referenceFile) : ""), [referenceFile]);

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      setRecords(await loadRecords("datasets", "/v1/datasets"));
    } catch (loadError) {
      setError(messageFrom(loadError));
    } finally {
      setLoading(false);
    }
  }

  async function saveDataset() {
    if (!okFiles.length || !faultFiles.length) {
      setError("Sube al menos un ejemplo correcto y uno incorrecto para registrar el dataset.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const [okImageUris, faultImageUris] = await Promise.all([
        uploadFiles(okFiles, selectedFamily, selectedZone, "dataset"),
        uploadFiles(faultFiles, selectedFamily, selectedZone, "dataset")
      ]);
      const payload = {
        name,
        family: selectedFamily,
        zone_id: selectedZone,
        ok_image_uris: okImageUris,
        fault_image_uris: faultImageUris,
        mask: maskMode === "auto" ? { type: "auto" } : {
          type: "polygon",
          points: maskPoints(maskRect)
        }
      };
      const dataset = await postJson("/v1/datasets/from-examples", payload);
      await insertSupabase("datasets", dataset as Record<string, unknown>);
      setOkFiles([]);
      setFaultFiles([]);
      setReferenceIndex(0);
      await refresh();
    } catch (saveError) {
      setError(messageFrom(saveError));
      setLoading(false);
    }
  }

  return (
    <>
      <header className="topbar">
        <div>
          <h1>Datasets</h1>
          <p>Sube ejemplos correctos e incorrectos. El sistema genera el manifest y la máscara internos.</p>
        </div>
        <button className="primary" onClick={refresh} disabled={loading}>Actualizar</button>
      </header>
      <section className="datasetGrid">
        <aside className="panel formPanel">
          <label>Nombre<input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <SelectField label="Familia" value={selectedFamily} onChange={setSelectedFamily} options={families.length ? families : [family]} />
          <SelectField label="Zona" value={selectedZone} onChange={setSelectedZone} options={zones.map((zone) => zone.id)} labels={Object.fromEntries(zones.map((zone) => [zone.id, zone.name || zone.label || zone.id]))} />
          <label>Ejemplos correctos<input type="file" accept="image/*" multiple onChange={(event) => setOkFiles(Array.from(event.target.files || []))} /></label>
          <label>Ejemplos incorrectos<input type="file" accept="image/*" multiple onChange={(event) => setFaultFiles(Array.from(event.target.files || []))} /></label>
          {okFiles.length ? (
            <label>Referencia correcta
              <select value={referenceIndex} onChange={(event) => setReferenceIndex(Number(event.target.value))}>
                {okFiles.map((file, index) => <option key={`${file.name}-${index}`} value={index}>{file.name}</option>)}
              </select>
            </label>
          ) : null}
          <label>Máscara
            <select value={maskMode} onChange={(event) => setMaskMode(event.target.value as "auto" | "manual")}>
              <option value="auto">Segmentar molde automáticamente</option>
              <option value="manual">Ajuste manual</option>
            </select>
          </label>
          <div className="datasetCounts">
            <span>{okFiles.length} correctas</span>
            <span>{faultFiles.length} incorrectas</span>
          </div>
          <p className="fieldNote">Manifest y máscara se generan automáticamente para entrenamiento.</p>
          <button className="primary fullWidth" onClick={saveDataset} disabled={loading}>Registrar dataset</button>
        </aside>
        <section className="panel maskPanel">
          <div className="captureHeader">
            <strong>Máscara de inspección</strong>
            <span className="status running">{maskMode === "auto" ? "segmentador" : "manual"}</span>
          </div>
          <div className="maskPreview">
            {referenceUrl ? <img src={referenceUrl} alt="Referencia correcta" /> : <span>Selecciona ejemplos correctos para ajustar la máscara</span>}
            {maskMode === "manual" ? (
              <div
                className="maskOverlay"
                style={{
                  left: `${maskRect.left}%`,
                  top: `${maskRect.top}%`,
                  width: `${maskRect.right - maskRect.left}%`,
                  height: `${maskRect.bottom - maskRect.top}%`
                }}
              />
            ) : <div className="autoMaskOverlay" />}
            <div className="frame"><i /><i /><i /><i /></div>
          </div>
          {maskMode === "manual" ? (
            <div className="maskControls">
              <RangeField label="Izquierda" value={maskRect.left} min={0} max={Math.min(maskRect.right - 5, 90)} onChange={(left) => setMaskRect((current) => ({ ...current, left }))} />
              <RangeField label="Derecha" value={maskRect.right} min={Math.max(maskRect.left + 5, 10)} max={100} onChange={(right) => setMaskRect((current) => ({ ...current, right }))} />
              <RangeField label="Arriba" value={maskRect.top} min={0} max={Math.min(maskRect.bottom - 5, 90)} onChange={(top) => setMaskRect((current) => ({ ...current, top }))} />
              <RangeField label="Abajo" value={maskRect.bottom} min={Math.max(maskRect.top + 5, 10)} max={100} onChange={(bottom) => setMaskRect((current) => ({ ...current, bottom }))} />
            </div>
          ) : <p className="fieldNote compact">La máscara se calcula con el segmentador genérico de molde antes del entrenamiento.</p>}
        </section>
        <RecordPanel loading={loading} error={error} records={records} hiddenKeys={["manifest_uri", "mask_uri", "dataset_uri", "output_uri"]} />
      </section>
    </>
  );
}

function ModelsView({ family, zoneId, zones }: { family: string; zoneId: string; zones: Zone[] }) {
  const [selectedFamily, setSelectedFamily] = useState(family);
  const [selectedZone, setSelectedZone] = useState(zoneId);
  const [records, setRecords] = useState<ResourceRecord[]>([]);
  const [datasets, setDatasets] = useState<ResourceRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const [loadedModels, loadedDatasets] = await Promise.all([
        loadRecords("model_versions", "/v1/model_versions"),
        loadRecords("datasets", "/v1/datasets")
      ]);
      setRecords(loadedModels);
      setDatasets(loadedDatasets);
    } catch (loadError) {
      setError(messageFrom(loadError));
    } finally {
      setLoading(false);
    }
  }

  async function startTraining() {
    const dataset = datasets.find((item) => fieldValue(item, "family") === selectedFamily && fieldValue(item, "zone_id") === selectedZone);
    if (!dataset) {
      setError("Registra primero un dataset para esta familia/zona.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      await postJson("/v1/training-jobs", {
        family: selectedFamily,
        zone_id: selectedZone,
        dataset_uri: fieldValue(dataset, "dataset_uri") || `system://${dataset.id}`,
        manifest_uri: fieldValue(dataset, "manifest_uri"),
        mask_uri: fieldValue(dataset, "mask_uri"),
        target: "cloud-gpu"
      });
      await refresh();
    } catch (trainingError) {
      setError(messageFrom(trainingError));
      setLoading(false);
    }
  }

  return (
    <>
      <header className="topbar">
        <div>
          <h1>Modelos</h1>
          <p>El entrenamiento evalúa candidatos y promueve automáticamente el modelo con mejor score por molde/zona.</p>
        </div>
        <button className="primary" onClick={refresh} disabled={loading}>Actualizar</button>
      </header>
      <section className="managementGrid">
        <aside className="panel formPanel">
          <label>Familia<input value={selectedFamily} onChange={(event) => setSelectedFamily(event.target.value)} /></label>
          <SelectField label="Zona" value={selectedZone} onChange={setSelectedZone} options={zones.map((zone) => zone.id)} labels={Object.fromEntries(zones.map((zone) => [zone.id, zone.name || zone.label || zone.id]))} />
          <p className="fieldNote">No se selecciona red neuronal manualmente. El backend prueba candidatos y guarda el ganador.</p>
          <button className="primary fullWidth" onClick={startTraining} disabled={loading}>Entrenar y autoelegir modelo</button>
        </aside>
        <RecordPanel loading={loading} error={error} records={records} hiddenKeys={["model_uri", "artifact_uri", "manifest_uri", "mask_uri"]} />
      </section>
    </>
  );
}

function BenchmarksView({ family, zoneId }: { family: string; zoneId: string }) {
  const [dataset, setDataset] = useState("mvtec_ad");
  const [category, setCategory] = useState("metal_nut");
  const [localRoot, setLocalRoot] = useState("");
  const [maxItems, setMaxItems] = useState(25);
  const [records, setRecords] = useState<ResourceRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      setRecords(await loadRecords("public_dataset_imports", "/v1/public_dataset_imports"));
    } catch (loadError) {
      setError(messageFrom(loadError));
    } finally {
      setLoading(false);
    }
  }

  async function importDataset() {
    setLoading(true);
    setError("");
    try {
      const record = await postJson("/v1/public-datasets/import", {
        dataset,
        category,
        local_root: localRoot || null,
        max_items: maxItems,
        family: `benchmark_${family}`,
        zone_id: zoneId
      });
      await insertSupabase("public_dataset_imports", record as Record<string, unknown>);
      await refresh();
    } catch (importError) {
      setError(messageFrom(importError));
      setLoading(false);
    }
  }

  return (
    <>
      <header className="topbar">
        <div>
          <h1>Benchmarks</h1>
          <p>Importa datasets públicos para validar la mecánica de entrenamiento, no para producción de moldes.</p>
        </div>
        <button className="primary" onClick={refresh} disabled={loading}>Actualizar</button>
      </header>
      <section className="managementGrid">
        <aside className="panel formPanel">
          <label>Dataset
            <select value={dataset} onChange={(event) => setDataset(event.target.value)}>
              <option value="mvtec_ad">MVTec AD</option>
              <option value="visa">VisA</option>
              <option value="kolektor_sdd">KolektorSDD</option>
              <option value="abo">ABO</option>
            </select>
          </label>
          <label>Categoría<input value={category} onChange={(event) => setCategory(event.target.value)} placeholder="metal_nut, screw, etc." /></label>
          <label>Ruta local descargada<input value={localRoot} onChange={(event) => setLocalRoot(event.target.value)} placeholder="/datasets/mvtec/metal_nut" /></label>
          <label>Máximo por clase<input type="number" min={1} max={500} value={maxItems} onChange={(event) => setMaxItems(Number(event.target.value))} /></label>
          <p className="fieldNote">Sin ruta local se registra la fuente/licencia y queda pendiente de descarga.</p>
          <button className="primary fullWidth" onClick={importDataset} disabled={loading}>Importar benchmark</button>
          {error ? <p className="errorText compactError">{error}</p> : null}
        </aside>
        <RecordPanel loading={loading} error="" records={records} hiddenKeys={["manifest_uri", "mask_uri"]} />
      </section>
    </>
  );
}

function ListView({ title, description, table, endpoint }: { title: string; description: string; table: string; endpoint: string }) {
  const [records, setRecords] = useState<ResourceRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      setRecords(await loadRecords(table, endpoint));
    } catch (loadError) {
      setError(messageFrom(loadError));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, [table, endpoint]);

  return (
    <>
      <header className="topbar">
        <div><h1>{title}</h1><p>{description}</p></div>
        <button className="primary" onClick={refresh} disabled={loading}>Actualizar</button>
      </header>
      <RecordPanel loading={loading} error={error} records={records} hiddenKeys={["manifest_uri", "mask_uri", "model_uri", "image_uri"]} />
    </>
  );
}

function ReferenceSetup({
  family,
  zoneId,
  reference,
  expectedPieces,
  onSaved
}: {
  family: string;
  zoneId: string;
  reference: ZoneReference | null;
  expectedPieces: ExpectedPiece[];
  onSaved: (reference: ZoneReference) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const referenceUrl = displayUrl(reference?.image_url || reference?.image_uri || "");

  async function saveReference() {
    if (!file) {
      setError("Sube foto referencia.");
      return;
    }
    setSaving(true);
    setError("");
    try {
      const imageUri = await uploadSystemFile(file, family, zoneId, "reference");
      const saved = (await postJson(`/v1/zones/${encodeURIComponent(zoneId)}/reference`, {
        family,
        image_uri: imageUri,
        reference_id: "golden_sample",
        tolerance: { translation: 0.08, scale: 0.2, rotation: 8, pose_score: 0.72 }
      })) as ZoneReference;
      onSaved(saved);
      setFile(null);
    } catch (saveError) {
      setError(messageFrom(saveError));
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="referenceStrip">
      <div className="referencePreview">
        {referenceUrl ? <img src={referenceUrl} alt="Golden sample" /> : <span>Sin golden sample</span>}
      </div>
      <div>
        <strong>Golden sample</strong>
        <span>{reference ? `${reference.family} / ${reference.zone_id} / ${reference.reference_id}` : "Primero carga foto correcta."}</span>
      </div>
      <div className="expectedChips">
        {(expectedPieces.length ? expectedPieces : defaultExpectedPieces()).slice(0, 6).map((piece) => <span key={piece.id}>{piece.name || piece.class_name}</span>)}
      </div>
      <label className="secondary fileButton">Subir referencia
        <input type="file" accept="image/*" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
      </label>
      <button className="primary" type="button" onClick={saveReference} disabled={saving || !file}>{saving ? "Guardando" : "Guardar"}</button>
      {error ? <p className="errorText compactError">{error}</p> : null}
    </section>
  );
}

function AnnotationWorkspace({
  family,
  zoneId,
  expectedPieces,
  reference
}: {
  family: string;
  zoneId: string;
  expectedPieces: ExpectedPiece[];
  reference: ZoneReference | null;
}) {
  const [selectedFamily, setSelectedFamily] = useState(family);
  const [selectedZone, setSelectedZone] = useState(zoneId);
  const [images, setImages] = useState<AnnotatableImage[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [boxes, setBoxes] = useState<AnnotationBox[]>([]);
  const [split, setSplit] = useState<"train" | "val" | "test">("train");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [dataset, setDataset] = useState<ResourceRecord | null>(null);
  const [job, setJob] = useState<ResourceRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const selectedImage = images.find((image) => image.id === selectedId) || images[0] || null;
  const pieces = expectedPieces.length ? expectedPieces : defaultExpectedPieces();

  useEffect(() => {
    setSelectedFamily(family);
    setSelectedZone(zoneId);
  }, [family, zoneId]);

  useEffect(() => {
    void refreshImages();
  }, [selectedFamily, selectedZone]);

  useEffect(() => {
    if (!selectedImage) {
      setBoxes([]);
      return;
    }
    setSelectedId(selectedImage.id);
    void loadImageAnnotations(selectedImage);
  }, [selectedImage?.id]);

  async function refreshImages() {
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const [dbRecords, backendRecords, annotationRecords] = await Promise.all([
        selectSupabase("inspections"),
        getJson("/v1/inspections").catch(() => []),
        getJson(`/v1/annotations?family=${encodeURIComponent(selectedFamily)}&zone_id=${encodeURIComponent(selectedZone)}`).catch(() => [])
      ]);
      const records = [
        ...(Array.isArray(annotationRecords) ? annotationRecords as ResourceRecord[] : []),
        ...dbRecords,
        ...(Array.isArray(backendRecords) ? backendRecords as ResourceRecord[] : [])
      ];
      const nextImages = records
        .map(recordToAnnotatableImage)
        .filter((image): image is AnnotatableImage => Boolean(image?.image_uri))
        .filter((image) => image.family === selectedFamily && image.zone_id === selectedZone);
      setImages(uniqueImages(nextImages));
      setSelectedId((current) => current || nextImages[0]?.id || "");
    } catch (loadError) {
      setError(messageFrom(loadError));
    } finally {
      setLoading(false);
    }
  }

  async function loadImageAnnotations(image: AnnotatableImage) {
    try {
      const records = await getJson(`/v1/annotations?image_uri=${encodeURIComponent(image.image_uri)}`);
      const latest = Array.isArray(records) && records.length ? records[records.length - 1] : null;
      const loaded = Array.isArray(latest?.annotations) ? latest.annotations.map(normalizeLoadedBox).filter(Boolean) as AnnotationBox[] : [];
      setBoxes(loaded);
      if (latest?.split) setSplit(latest.split);
    } catch {
      setBoxes([]);
    }
  }

  async function uploadAnnotationImage() {
    if (!uploadFile) {
      setError("Sube una foto.");
      return;
    }
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const imageUri = await uploadSystemFile(uploadFile, selectedFamily, selectedZone, "annotation");
      const image = {
        id: imageIdFromUri(imageUri),
        image_uri: imageUri,
        image_url: displayUrl(imageUri),
        family: selectedFamily,
        zone_id: selectedZone,
      };
      setImages((current) => uniqueImages([image, ...current]));
      setSelectedId(image.id);
      setUploadFile(null);
    } catch (uploadError) {
      setError(messageFrom(uploadError));
    } finally {
      setLoading(false);
    }
  }

  async function saveBoxes() {
    if (!selectedImage) {
      setError("No hay imagen.");
      return;
    }
    if (!boxes.length) {
      setError("Dibuja al menos una caja.");
      return;
    }
    setLoading(true);
    setError("");
    setNotice("");
    try {
      await postJson("/v1/annotations", {
        image_id: selectedImage.id,
        image_uri: selectedImage.image_uri,
        family: selectedFamily,
        zone_id: selectedZone,
        mold_id: selectedImage.mold_id || null,
        session_id: selectedImage.session_id || null,
        reference_id: reference?.reference_id || null,
        split,
        annotations: boxes
      });
      await loadImageAnnotations(selectedImage);
    } catch (saveError) {
      setError(messageFrom(saveError));
    } finally {
      setLoading(false);
    }
  }

  async function exportDataset() {
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const created = (await postJson("/v1/segmenter-datasets/from-annotations", {
        family: selectedFamily,
        zone_id: selectedZone,
        name: `YOLO ${selectedFamily} ${selectedZone}`
      })) as ResourceRecord;
      setDataset(created);
    } catch (datasetError) {
      setError(messageFrom(datasetError));
    } finally {
      setLoading(false);
    }
  }

  async function trainInspector() {
    if (!dataset?.id) {
      setError("Primero genera dataset.");
      return;
    }
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const created = (await postJson("/v1/inspector-training-jobs", {
        family: selectedFamily,
        zone_id: selectedZone,
        dataset_id: dataset.id,
        target: "cloud-gpu"
      })) as ResourceRecord;
      setJob(created);
    } catch (trainError) {
      setError(messageFrom(trainError));
    } finally {
      setLoading(false);
    }
  }

  async function autoAnnotateDrafts() {
    if (!selectedImage) {
      setError("Selecciona una imagen.");
      return;
    }
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const response = await postJson("/v1/annotations/auto-draft", {
        family: selectedFamily,
        zone_id: selectedZone,
        image_uri: selectedImage.image_uri
      }) as { source?: string; message?: string; annotations?: unknown[] };
      const drafts = Array.isArray(response.annotations) ? response.annotations.map(normalizeLoadedBox).filter(Boolean) as AnnotationBox[] : [];
      if (!drafts.length) {
        const localDrafts = pieces.map(pieceToDraftBox).filter((box): box is AnnotationBox => Boolean(box));
        if (!localDrafts.length) {
          setError(response.message || "No hay modelo, anotaciones previas ni ROIs para auto-anotar esta zona.");
          return;
        }
        mergeDraftBoxes(localDrafts);
        setNotice("Borrador local desde ROIs. Corrige antes de guardar.");
        return;
      }
      mergeDraftBoxes(drafts);
      setNotice(response.message || "Borrador creado. Corrige antes de guardar.");
    } catch (draftError) {
      const localDrafts = pieces.map(pieceToDraftBox).filter((box): box is AnnotationBox => Boolean(box));
      if (!localDrafts.length) {
        setError(messageFrom(draftError));
        return;
      }
      mergeDraftBoxes(localDrafts);
      setNotice("Borrador local desde ROIs. Corrige antes de guardar.");
    } finally {
      setLoading(false);
    }
  }

  function mergeDraftBoxes(drafts: AnnotationBox[]) {
    setBoxes((current) => {
      const existing = new Set(current.map((box) => box.element_id || box.class_name));
      return [...current, ...drafts.filter((box) => !existing.has(box.element_id || box.class_name))];
    });
  }

  return (
    <>
      <header className="topbar">
        <div>
          <h1>Anotar piezas</h1>
          <p>Dibuja cajas sobre piezas visibles y genera dataset YOLO.</p>
        </div>
        <button className="primary" type="button" onClick={refreshImages} disabled={loading}>Actualizar</button>
      </header>

      {error ? <p className="errorText inlineError">{error}</p> : null}
      {notice ? <p className="infoText inlineError">{notice}</p> : null}

      <section className="annotationGrid">
        <aside className="panel annotationSidebar">
          <label>Familia<input value={selectedFamily} onChange={(event) => setSelectedFamily(event.target.value)} /></label>
          <label>Zona<input value={selectedZone} onChange={(event) => setSelectedZone(event.target.value)} /></label>
          <label>Split
            <select value={split} onChange={(event) => setSplit(event.target.value as "train" | "val" | "test")}>
              <option value="train">train</option>
              <option value="val">val</option>
              <option value="test">test</option>
            </select>
          </label>
          <label className="fileTile">
            <span>Subir imagen</span>
            <strong>{uploadFile ? uploadFile.name : "Seleccionar"}</strong>
            <small>Foto horizontal del molde</small>
            <input type="file" accept="image/*" onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)} />
          </label>
          <button className="secondary fullWidth" type="button" onClick={uploadAnnotationImage} disabled={loading || !uploadFile}>Agregar imagen</button>
          <div className="annotationImageList">
            {images.length ? images.map((image) => (
              <button key={image.id} className={selectedImage?.id === image.id ? "active" : ""} type="button" onClick={() => setSelectedId(image.id)}>
                <strong>{image.id}</strong>
                <span>{image.created_at ? formatDate(image.created_at) : image.zone_id}</span>
              </button>
            )) : <span className="emptyText compact">Sin imágenes.</span>}
          </div>
        </aside>

        <section className="panel annotationPanel">
          <ImageAnnotator image={selectedImage} pieces={pieces} boxes={boxes} onChange={setBoxes} />
          <div className="annotationActions">
            <button className="secondary" type="button" onClick={autoAnnotateDrafts} disabled={loading || !selectedImage}>Auto-anotar borrador</button>
            <button className="primary" type="button" onClick={saveBoxes} disabled={loading || !selectedImage || !boxes.length}>Guardar anotaciones</button>
            <button className="secondary" type="button" onClick={exportDataset} disabled={loading}>Generar dataset YOLO</button>
            <button className="secondary" type="button" onClick={trainInspector} disabled={loading || !dataset}>Entrenar inspector</button>
          </div>
        </section>

        <aside className="panel annotationSidebar">
          <strong>Estado</strong>
          <Metric label="Cajas" value={String(boxes.length)} />
          <Metric label="Dataset" value={dataset?.id ? String(dataset.id) : "Pendiente"} />
          <Metric label="Training" value={job?.id ? String(job.id) : "Pendiente"} />
          <div className="workflowNote">
            <strong>Proceso</strong>
            <span>Sube imagen, auto-anota como borrador, corrige cajas y guarda. Luego genera dataset y entrena inspector.</span>
          </div>
          <div className="expectedChips block">
            {pieces.map((piece) => <span key={piece.id}>{piece.name || piece.class_name}</span>)}
          </div>
        </aside>
      </section>
    </>
  );
}

function ImageAnnotator({
  image,
  pieces,
  boxes,
  onChange
}: {
  image: AnnotatableImage | null;
  pieces: ExpectedPiece[];
  boxes: AnnotationBox[];
  onChange: (boxes: AnnotationBox[]) => void;
}) {
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const [className, setClassName] = useState(pieces[0]?.class_name || "piece");
  const [draft, setDraft] = useState<[number, number, number, number] | null>(null);
  const [drawing, setDrawing] = useState(false);

  useEffect(() => {
    if (pieces[0]?.class_name && !pieces.some((piece) => piece.class_name === className)) {
      setClassName(pieces[0].class_name);
    }
  }, [pieces, className]);

  function point(event: React.PointerEvent<HTMLDivElement>) {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return [0, 0] as const;
    return [
      clamp((event.clientX - rect.left) / rect.width, 0, 1),
      clamp((event.clientY - rect.top) / rect.height, 0, 1)
    ] as const;
  }

  function start(event: React.PointerEvent<HTMLDivElement>) {
    if (!image) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const [x, y] = point(event);
    setDraft([x, y, x, y]);
    setDrawing(true);
  }

  function move(event: React.PointerEvent<HTMLDivElement>) {
    if (!drawing || !draft) return;
    const [x, y] = point(event);
    setDraft([draft[0], draft[1], x, y]);
  }

  function end() {
    if (!drawing || !draft) return;
    const box = normalizeBox(draft);
    setDrawing(false);
    setDraft(null);
    if (box[2] - box[0] < 0.01 || box[3] - box[1] < 0.01) return;
    const piece = pieces.find((item) => item.class_name === className);
    onChange([
      ...boxes,
      {
        id: `box_${Date.now().toString(36)}`,
        element_id: piece?.id || className,
        class_name: className,
        bbox: box,
        status: "present"
      }
    ]);
  }

  return (
    <div className="imageAnnotator">
      <div className="annotatorToolbar">
        <label>Etiqueta
          <select value={className} onChange={(event) => setClassName(event.target.value)}>
            {pieces.map((piece) => <option key={piece.id} value={piece.class_name}>{piece.name || piece.class_name}</option>)}
          </select>
        </label>
      </div>
      <div ref={canvasRef} className="annotatorCanvas" onPointerDown={start} onPointerMove={move} onPointerUp={end} onPointerCancel={end}>
        {image ? <img src={displayUrl(image.image_url || image.image_uri)} alt="Imagen para anotar" draggable={false} /> : <span>Selecciona o sube imagen.</span>}
        <svg className="annotationOverlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          {boxes.map((box) => <AnnotationRect key={box.id} box={box} />)}
          {draft ? <AnnotationRect box={{ id: "draft", class_name: className, bbox: normalizeBox(draft), status: "present" }} draft /> : null}
        </svg>
      </div>
      <div className="boxList">
        {boxes.map((box) => (
          <div key={box.id}>
            <select value={box.class_name} onChange={(event) => onChange(boxes.map((item) => item.id === box.id ? { ...item, class_name: event.target.value } : item))}>
              {pieces.map((piece) => <option key={piece.id} value={piece.class_name}>{piece.name || piece.class_name}</option>)}
            </select>
            <select value={box.status} onChange={(event) => onChange(boxes.map((item) => item.id === box.id ? { ...item, status: event.target.value as AnnotationBox["status"] } : item))}>
              <option value="present">present</option>
              <option value="missing">missing</option>
              <option value="uncertain">uncertain</option>
            </select>
            <button className="dangerButton" type="button" onClick={() => onChange(boxes.filter((item) => item.id !== box.id))}>Borrar</button>
          </div>
        ))}
      </div>
    </div>
  );
}

function AnnotationRect({ box, draft = false }: { box: AnnotationBox; draft?: boolean }) {
  const [x1, y1, x2, y2] = box.bbox;
  return <rect className={draft ? "draft" : ""} x={x1 * 100} y={y1 * 100} width={(x2 - x1) * 100} height={(y2 - y1) * 100} />;
}

function CaptureStillPanel({
  previewUrl,
  referenceUrl,
  status,
  onOpen,
  onFile,
  onInspect
}: {
  previewUrl: string;
  referenceUrl: string;
  status: InspectionStatus;
  onOpen: () => void;
  onFile: (file: File | null) => void;
  onInspect: () => void;
}) {
  return (
    <div className="captureStill">
      <div className="stillPreview">
        {previewUrl ? <img src={previewUrl} alt="Foto capturada" /> : <span>Sin foto capturada</span>}
        {!previewUrl && referenceUrl ? <img className="referenceGhost" src={referenceUrl} alt="Referencia golden sample" /> : null}
        <div className="frame"><i /><i /><i /><i /></div>
      </div>
      <div className="cameraActions">
        <label className="secondary fileButton">Subir foto
          <input type="file" accept="image/*" capture="environment" onChange={(event) => onFile(event.target.files?.[0] ?? null)} />
        </label>
        {previewUrl ? <button className="secondary" type="button" onClick={() => onInspect()} disabled={status === "uploading" || status === "running"}>Validar foto</button> : null}
        <button className="primary" type="button" onClick={onOpen} disabled={status === "uploading" || status === "running"}>Capturar foto</button>
      </div>
    </div>
  );
}

function CameraCapture({
  family,
  zoneId,
  previewUrl,
  reference,
  expectedPieces,
  onFile,
  onCapture,
  onCancel,
  fullscreen = false
}: {
  family: string;
  zoneId: string;
  previewUrl: string;
  reference?: ZoneReference | null;
  expectedPieces?: ExpectedPiece[];
  onFile: (file: File | null) => void;
  onCapture?: (file: File) => void | Promise<void>;
  onCancel?: () => void;
  fullscreen?: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const focusMaskId = useRef(`zone-focus-${Math.random().toString(36).slice(2)}`);
  const guidancePending = useRef(false);
  const stableReadyFrames = useRef(0);
  const autoCapturePending = useRef(false);
  const [active, setActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [guidance, setGuidance] = useState<CaptureGuidanceResult | null>(null);
  const [readyFrames, setReadyFrames] = useState(0);
  const [overlayOpacity, setOverlayOpacity] = useState(0.48);

  useEffect(() => {
    return () => stopCamera(streamRef);
  }, []);

  useEffect(() => {
    if (fullscreen && !active && !streamRef.current) {
      void startCamera();
    }
  }, [fullscreen]);

  useEffect(() => {
    const video = videoRef.current;
    const stream = streamRef.current;
    if (!active || !video || !stream) return;
    if (video.srcObject !== stream) {
      video.srcObject = stream;
    }
    void video.play().catch((error) => setCameraError(cameraErrorMessage(error)));
  }, [active]);

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => {
      void requestGuidance();
    }, 1400);
    return () => window.clearInterval(timer);
  }, [active, family, zoneId]);

  async function startCamera() {
    setCameraError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Cámara no disponible en este navegador.");
      return;
    }
    try {
      const stream = await openCameraStream();
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setActive(true);
    } catch (error) {
      setCameraError(cameraErrorMessage(error));
    }
  }

  async function openCameraStream() {
    const attempts: MediaStreamConstraints[] = [
      { video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false },
      { video: { facingMode: "environment" }, audio: false },
      { video: true, audio: false }
    ];
    let lastError: unknown = null;
    for (const constraints of attempts) {
      try {
        return await navigator.mediaDevices.getUserMedia(constraints);
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError;
  }

  function stop() {
    stopCamera(streamRef);
    setActive(false);
    stableReadyFrames.current = 0;
    setReadyFrames(0);
  }

  function closeFullscreen() {
    stop();
    onCancel?.();
  }

  async function capturePhoto() {
    if (autoCapturePending.current) return;
    autoCapturePending.current = true;
    const file = await frameFile("captura.jpg");
    if (file) {
      onFile(file);
      stop();
      await onCapture?.(file);
    }
    autoCapturePending.current = false;
  }

  async function requestGuidance() {
    if (!active || guidancePending.current) return;
    const file = await frameFile("guidance.jpg", 0.72);
    if (!file) return;
    guidancePending.current = true;
    try {
      const imageUri = await uploadSystemFile(file, family, zoneId, "inspection");
      const response = (await postJson("/v1/capture-guidance", { family, zone_id: zoneId, image_uri: imageUri, reference_id: reference?.reference_id || null })) as CaptureGuidanceResult;
      setGuidance(response);
      stableReadyFrames.current = response.auto_capture_ready ? stableReadyFrames.current + 1 : 0;
      setReadyFrames(stableReadyFrames.current);
      if (stableReadyFrames.current >= 3) {
        await capturePhoto();
      }
    } catch (error) {
      setGuidance({ ok: false, auto_capture_ready: false, message: messageFrom(error), guidance: [messageFrom(error)], quality: {}, alignment: {} });
      stableReadyFrames.current = 0;
      setReadyFrames(0);
    } finally {
      guidancePending.current = false;
    }
  }

  async function frameFile(filename: string, quality = 0.9): Promise<File | null> {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return null;
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
    return blob ? new File([blob], filename, { type: "image/jpeg" }) : null;
  }

  const blockedByGuidance = false;
  const confidence = guidance?.alignment?.mold_segmentation?.confidence;
  const livePolygon = guidance?.alignment?.mold_segmentation?.polygon_normalized;
  const guidanceItems = guidance?.guidance?.length ? guidance.guidance : ["Centra el molde dentro del marco."];
  const readiness = Math.min(3, readyFrames);
  const referenceUrl = displayUrl(reference?.image_url || reference?.image_uri || "");
  const guideRois = (expectedPieces || []).filter((piece) => Array.isArray(piece.roi) && piece.roi.length === 4).slice(0, 8);
  const focusRects = guideRois.length ? guideRois.map((piece) => ({
    id: piece.id,
    roi: piece.roi as number[],
    label: piece.name || piece.class_name
  })) : [{ id: "main-zone", roi: [0.18, 0.18, 0.82, 0.82], label: "zona" }];

  const stage = (
    <div className="dropzone cameraStage">
      {active || fullscreen ? <video ref={videoRef} playsInline muted autoPlay /> : previewUrl ? <img src={previewUrl} alt="Vista previa" /> : <span>Toma o sube una foto de la zona</span>}
      {fullscreen && cameraError ? (
        <div className="cameraStatusOverlay" role="status">
          <CircleAlert size={22} />
          <strong>{cameraError}</strong>
          <span>Revisa permiso o cámara conectada.</span>
        </div>
      ) : null}
      {!fullscreen ? <input type="file" accept="image/*" capture="environment" onChange={(event) => onFile(event.target.files?.[0] ?? null)} /> : null}
      {active && referenceUrl && !fullscreen ? <img className="referenceOverlayImage" src={referenceUrl} alt="Referencia golden sample" style={{ opacity: overlayOpacity }} /> : null}
      {fullscreen ? (
        <svg className="zoneFocusMask" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <mask id={focusMaskId.current}>
              <rect x="0" y="0" width="100" height="100" fill="white" />
              {focusRects.map((rect) => (
                <rect
                  key={rect.id}
                  x={rect.roi[0] * 100}
                  y={rect.roi[1] * 100}
                  width={(rect.roi[2] - rect.roi[0]) * 100}
                  height={(rect.roi[3] - rect.roi[1]) * 100}
                  rx="1.2"
                  fill="black"
                />
              ))}
            </mask>
          </defs>
          <rect className="zoneFocusShade" x="0" y="0" width="100" height="100" mask={`url(#${focusMaskId.current})`} />
          {focusRects.map((rect) => (
            <rect
              key={rect.id}
              className="zoneFocusWindow"
              x={rect.roi[0] * 100}
              y={rect.roi[1] * 100}
              width={(rect.roi[2] - rect.roi[0]) * 100}
              height={(rect.roi[3] - rect.roi[1]) * 100}
              rx="1.2"
            />
          ))}
        </svg>
      ) : null}
      {active && livePolygon?.length ? (
        <svg className="liveMoldPolygon" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <polygon points={polygonPoints(livePolygon)} />
        </svg>
      ) : null}
      {active && guideRois.length ? (
        <svg className="expectedGuideOverlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          {guideRois.map((piece) => <rect key={piece.id} x={(piece.roi?.[0] || 0) * 100} y={(piece.roi?.[1] || 0) * 100} width={((piece.roi?.[2] || 0) - (piece.roi?.[0] || 0)) * 100} height={((piece.roi?.[3] || 0) - (piece.roi?.[1] || 0)) * 100} />)}
        </svg>
      ) : null}
      <div className="frame"><i /><i /><i /><i /></div>
    </div>
  );

  if (fullscreen) {
    return (
      <div className="fullscreenCamera" role="dialog" aria-modal="true" aria-label="Captura guiada">
        <div className="fullscreenTop">
          <div>
            <strong>Captura guiada</strong>
            <span>{reference ? `Referencia ${reference.reference_id}` : "Carga una referencia para bloquear el ángulo correcto."}</span>
          </div>
          <button className="secondary" type="button" onClick={closeFullscreen}>Cancelar</button>
        </div>
        <div className="fullscreenCaptureGrid">
          <div className="fullscreenStageWrap">
            {stage}
            <div className="cameraFloatingActions" aria-label="Acciones de captura">
              <button className="primary captureTrigger" type="button" onClick={capturePhoto} disabled={Boolean(!active || cameraError)} aria-label="Capturar" title="Capturar">
                <Camera size={22} />
              </button>
              {cameraError ? (
                <button className="secondary retryCameraButton" type="button" onClick={startCamera}>
                  Reintentar
                </button>
              ) : null}
            </div>
          </div>
        </div>
        <div className={`fullscreenGuidance ${guidance?.ok ? "isOk" : cameraError ? "hasError" : "needsWork"}`}>
          <div>
            <strong>{cameraError || guidance?.message || (active ? "Detectando molde..." : "Abriendo cámara...")}</strong>
            <span>{typeof confidence === "number" ? `Confianza ${Math.round(confidence * 100)}%` : active ? "Cámara activa" : "Esperando cámara"}</span>
          </div>
          <div className="readinessMeter" aria-label="estabilidad">
            {[0, 1, 2].map((item) => <i key={item} className={item < readiness ? "ready" : ""} />)}
          </div>
          <ul>{guidanceItems.map((item) => <li key={item}>{item}</li>)}</ul>
        </div>
        <canvas ref={canvasRef} hidden />
      </div>
    );
  }

  return (
    <div className="cameraBox">
      {stage}
      <canvas ref={canvasRef} hidden />
      <div className="cameraActions">
        {active ? (
          <>
            <button className="secondary" type="button" onClick={stop}>Detener cámara</button>
            <button className="primary" type="button" onClick={capturePhoto} disabled={Boolean(blockedByGuidance)}>Capturar</button>
          </>
        ) : (
          <button className="secondary" type="button" onClick={startCamera}>Abrir cámara</button>
        )}
      </div>
      {active || guidance || cameraError ? (
        <div className={`guidanceBox ${guidance?.ok ? "isOk" : "needsWork"}`}>
          <strong>{cameraError || guidance?.message || "Preparando guía de captura..."}</strong>
          {typeof confidence === "number" ? <span>Confianza molde: {Math.round(confidence * 100)}%</span> : null}
          {guidance?.auto_capture_ready ? <span>Auto-captura lista</span> : null}
          <ul>{guidanceItems.map((item) => <li key={item}>{item}</li>)}</ul>
        </div>
      ) : null}
    </div>
  );
}

function InspectionResultPanel({
  status,
  message,
  result,
  previewUrl,
  family,
  zoneId
}: {
  status: InspectionStatus;
  message: string;
  result: InspectionResult | null;
  previewUrl: string;
  family: string;
  zoneId: string;
}) {
  const missingPolygons = missingPiecePolygons(result);
  const missingCount = result?.result?.piece_inspection?.missing_count ?? missingPolygons.length;
  const resultImage = inspectionImageUrl(result) || previewUrl;
  const identifiedModel = result?.result?.model_id || result?.result?.model_version || `${family} / ${zoneId}`;

  return (
    <section className="panel resultPanel resultVertical">
      <div className="resultSummary">
        <span className={`resultBadge ${status}`}>{statusLabel(status)}</span>
        <div>
          <h2>{message}</h2>
          <p>Molde identificado: <strong>{identifiedModel}</strong></p>
        </div>
      </div>
      <EvaluationImage imageUrl={resultImage} missingPolygons={missingPolygons} />
      <div className="metrics resultMetrics">
        <Metric label="Score" value={formatNumber(result?.result?.anomaly_score)} />
        <Metric label="Piezas faltantes" value={String(missingCount)} />
        <Metric label="Confianza molde" value={formatPercent(result?.result?.mold_segmentation?.confidence)} />
      </div>
      <div className="resultGuidance">
        <ul>{(result?.guidance?.length ? result.guidance : ["La evaluación aparecerá después de la auto-captura."]).map((item) => <li key={item}>{item}</li>)}</ul>
      </div>
    </section>
  );
}

function EvaluationImage({ imageUrl, missingPolygons }: { imageUrl: string; missingPolygons: Point[][] }) {
  return (
    <div className="evaluationImage">
      {imageUrl ? <img src={imageUrl} alt="Imagen evaluada" /> : <span>La imagen evaluada aparecerá aquí.</span>}
      {imageUrl && missingPolygons.length ? (
        <svg className="missingPieceOverlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          {missingPolygons.map((polygon, index) => <polygon key={index} points={polygonPoints(polygon)} />)}
        </svg>
      ) : null}
    </div>
  );
}

function missingPiecePolygons(result: InspectionResult | null) {
  const findings = result?.result?.piece_inspection?.findings || [];
  return findings
    .filter((finding) => finding.status === "missing")
    .map((finding) => finding.region || finding.polygon_normalized || finding.polygon || bboxToPolygon(finding.bbox_normalized))
    .filter((polygon): polygon is Point[] => Boolean(polygon?.length));
}

function inspectionImageUrl(result: InspectionResult | null) {
  const uri = result?.evidence?.overlay_image || result?.overlay_image_uri || "";
  return uri ? resolveUrl(uri) : "";
}

function bboxToPolygon(bbox?: { x: number; y: number; width: number; height: number }) {
  if (!bbox) return null;
  return [
    { x: bbox.x, y: bbox.y },
    { x: bbox.x + bbox.width, y: bbox.y },
    { x: bbox.x + bbox.width, y: bbox.y + bbox.height },
    { x: bbox.x, y: bbox.y + bbox.height }
  ];
}

function RangeField({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (value: number) => void }) {
  return (
    <label>{label}
      <input type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function SelectField({ label, value, onChange, options, labels = {} }: { label: string; value: string; onChange: (value: string) => void; options: string[]; labels?: Record<string, string> }) {
  return (
    <label>{label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {Array.from(new Set(options)).map((option) => <option key={option} value={option}>{labels[option] || option}</option>)}
      </select>
    </label>
  );
}

function FilePicker({
  label,
  detail,
  count,
  multiple = false,
  onChange
}: {
  label: string;
  detail: string;
  count: number;
  multiple?: boolean;
  onChange: (files: File[]) => void;
}) {
  return (
    <label className="fileTile">
      <span>{label}</span>
      <strong>{count ? `${count} archivo${count === 1 ? "" : "s"}` : "Seleccionar"}</strong>
      <small>{detail}</small>
      <input type="file" accept="image/*" multiple={multiple} onChange={(event) => onChange(Array.from(event.target.files || []))} />
    </label>
  );
}

function RecordPanel({ loading, error, records, hiddenKeys }: { loading: boolean; error: string; records: ResourceRecord[]; hiddenKeys: string[] }) {
  return (
    <section className="panel tablePanel">
      <div className="captureHeader">
        <strong>{loading ? "Cargando..." : `${records.length} registros`}</strong>
        {error ? <span className="status error">error</span> : <span className="status correct">api/db</span>}
      </div>
      {error ? <p className="errorText">{error}</p> : records.length ? (
        <div className="recordList">
          {records.map((record) => (
            <article key={String(record.id)} className="recordRow">
              <strong>{String(record.id)}</strong>
              <pre>{JSON.stringify(publicRecord(record, hiddenKeys), null, 2)}</pre>
            </article>
          ))}
        </div>
      ) : <p className="emptyText">Sin registros todavía.</p>}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metricTile"><span>{label}</span><strong>{value}</strong></div>;
}

async function loadCaptureContext(family: string, zoneId: string): Promise<{ reference: ZoneReference | null; expected: ExpectedPiece[] }> {
  const [reference, expected] = await Promise.all([
    loadZoneReference(family, zoneId),
    getJson(`/v1/zones/${encodeURIComponent(zoneId)}/expected?family=${encodeURIComponent(family)}`).catch(() => [])
  ]);
  return {
    reference: reference as ZoneReference | null,
    expected: Array.isArray(expected) && expected.length ? expected as ExpectedPiece[] : defaultExpectedPieces()
  };
}

async function loadRemoteSectionPlan(family: string, moldKey: string, seedZoneId: string): Promise<MoldSectionPlan | null> {
  const records = await getJson(`/v1/mold-section-plans?family=${encodeURIComponent(family)}`).catch(() => []);
  const record = Array.isArray(records)
    ? records.find((item) => {
      const source = recordSource(item as ResourceRecord);
      return String(source.mold_key || source.moldKey || "") === (moldKey || family);
    }) || null
    : null;
  return sectionPlanFromRecord(record, family, moldKey, seedZoneId);
}

async function loadZoneReference(family: string, zoneId: string): Promise<ZoneReference | null> {
  const records = await getJson("/v1/references").catch(() => []);
  if (!Array.isArray(records)) return null;
  const matches = records
    .map((record) => recordSource(record as ResourceRecord))
    .filter((record) => String(record.family || "") === family && String(record.zone_id || record.zoneId || "") === zoneId);
  if (!matches.length) return null;
  const latest = matches.sort((left, right) => Date.parse(String(right.updated_at || right.updatedAt || right.created_at || "")) - Date.parse(String(left.updated_at || left.updatedAt || left.created_at || "")))[0];
  return latest as ZoneReference;
}

async function persistRemoteSectionPlan(plan: MoldSectionPlan): Promise<MoldSectionPlan | null> {
  const record = await postJson(`/v1/mold-section-plans/${encodeURIComponent(plan.moldKey || plan.family)}`, {
    family: plan.family,
    mold_key: plan.moldKey || plan.family,
    source: "manual",
    sections: plan.sections.map((section, index) => ({
      id: section.id,
      zone_id: section.zoneId,
      label: section.label,
      zone_index: section.zoneIndex,
      view: section.view,
      required: true,
      order: index + 1
    }))
  }).catch(() => null);
  return sectionPlanFromRecord(record, plan.family, plan.moldKey, plan.sections[0]?.zoneId || DEFAULT_ZONES[0].id);
}

async function ensureValidationSession(plan: MoldSectionPlan): Promise<MoldValidationSession | null> {
  if (!plan.sections.length) return null;
  await persistRemoteSectionPlan(plan);
  const session = await postJson("/v1/mold-validation-sessions", {
    family: plan.family,
    mold_key: plan.moldKey || plan.family
  }).catch(() => null);
  return normalizeValidationSession(session);
}

async function recordValidationSectionResult(
  session: MoldValidationSession | null,
  section: MoldSection | null,
  payload: {
    status: "correct" | "review" | "retake_photo";
    inspection_id?: string;
    image_uri?: string;
    message?: string;
  }
): Promise<MoldValidationSession | null> {
  if (!session || !section) return null;
  const updated = await postJson(`/v1/mold-validation-sessions/${encodeURIComponent(session.id)}/sections/${encodeURIComponent(section.id)}`, {
    section_id: section.id,
    zone_id: section.zoneId,
    ...payload
  }).catch(() => null);
  return normalizeValidationSession(updated);
}

function normalizeValidationSession(value: unknown): MoldValidationSession | null {
  if (!value || typeof value !== "object") return null;
  const source = value as Record<string, unknown>;
  const status = source.status === "complete" || source.status === "in_progress" ? source.status : "pending";
  return {
    id: String(source.id || ""),
    family: String(source.family || ""),
    mold_key: source.mold_key ? String(source.mold_key) : undefined,
    status,
    required_count: Number(source.required_count || 0),
    completed_count: Number(source.completed_count || 0),
    missing_section_ids: Array.isArray(source.missing_section_ids) ? source.missing_section_ids.map(String) : [],
    ready_section_ids: Array.isArray(source.ready_section_ids) ? source.ready_section_ids.map(String) : [],
    section_results: source.section_results && typeof source.section_results === "object" ? source.section_results as Record<string, Record<string, unknown>> : {}
  };
}

function validationResultsFromSession(session: MoldValidationSession): Record<string, SectionResult> {
  const next: Record<string, SectionResult> = {};
  for (const [sectionId, value] of Object.entries(session.section_results || {})) {
    const status = value.status === "correct" || value.status === "review" || value.status === "retake_photo" ? value.status : "review";
    next[sectionId] = {
      status,
      message: value.message ? String(value.message) : validationStatusLabel(status === "retake_photo" ? "in_progress" : session.status),
      updatedAt: String(value.updated_at || new Date().toISOString()),
      inspectionId: value.inspection_id ? String(value.inspection_id) : null,
      imageUri: value.image_uri ? String(value.image_uri) : null
    };
  }
  return next;
}

function sectionPlanFromRecord(record: unknown, fallbackFamily: string, fallbackMoldKey: string, seedZoneId: string): MoldSectionPlan | null {
  if (!record || typeof record !== "object") return null;
  const source = recordSource(record as ResourceRecord);
  const rawSections = source.sections;
  if (!Array.isArray(rawSections) || !rawSections.length) return null;
  const sections = rawSections.map(normalizeSection).filter(Boolean) as MoldSection[];
  if (!sections.length) return null;
  return {
    family: String(source.family || fallbackFamily),
    moldKey: String(source.mold_key || source.moldKey || fallbackMoldKey || fallbackFamily),
    sections,
    updatedAt: String(source.updated_at || source.updatedAt || new Date().toISOString())
  };
}

function defaultExpectedPieces(): ExpectedPiece[] {
  return [
    { id: "guide_post", class_name: "guide_post", name: "Poste guía", required: true, critical: true },
    { id: "insert_block", class_name: "insert_block", name: "Bloque inserto", required: true, critical: true },
    { id: "black_fastener", class_name: "black_fastener", name: "Perno negro", required: true, critical: true },
    { id: "yellow_guide", class_name: "yellow_guide", name: "Guía amarilla", required: true, critical: true },
    { id: "round_bushing", class_name: "round_bushing", name: "Buje circular", required: true, critical: true }
  ];
}

function pieceToDraftBox(piece: ExpectedPiece): AnnotationBox | null {
  if (!Array.isArray(piece.roi) || piece.roi.length !== 4) return null;
  return {
    id: `auto_${piece.id}_${Date.now().toString(36)}`,
    element_id: piece.id,
    class_name: piece.class_name,
    bbox: normalizeBox(piece.roi as [number, number, number, number]),
    status: "present",
    notes: "auto_draft_roi"
  };
}

async function validateClientImage(file: File) {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  const maxWidth = 320;
  const scale = Math.min(1, maxWidth / bitmap.width);
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return { ok: true, message: "Foto lista.", brightness: 0, blur: 0 };
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const gray = new Float32Array(canvas.width * canvas.height);
  let total = 0;
  for (let index = 0, pixel = 0; index < data.length; index += 4, pixel += 1) {
    const luminance = data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114;
    gray[pixel] = luminance;
    total += luminance;
  }
  const brightness = total / gray.length;
  const blur = laplacianVariance(gray, canvas.width, canvas.height);
  const tooDark = brightness < 42;
  const tooBright = brightness > 232;
  const tooBlurred = blur < 18;
  if (tooDark || tooBright || tooBlurred) {
    return {
      ok: false,
      message: tooBlurred ? "Retoma foto: está borrosa." : "Retoma foto: luz fuera de rango.",
      brightness: Math.round(brightness),
      blur: Math.round(blur)
    };
  }
  return { ok: true, message: "Foto lista.", brightness: Math.round(brightness), blur: Math.round(blur) };
}

function laplacianVariance(gray: Float32Array, width: number, height: number) {
  const values: number[] = [];
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const center = gray[y * width + x] * 4;
      const value = center - gray[y * width + x - 1] - gray[y * width + x + 1] - gray[(y - 1) * width + x] - gray[(y + 1) * width + x];
      values.push(value);
    }
  }
  if (!values.length) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
}

function recordToAnnotatableImage(record: ResourceRecord): AnnotatableImage | null {
  const source = recordSource(record);
  const imageUri = String(source.image_uri || "");
  if (!imageUri) return null;
  return {
    id: String(source.image_id || source.id || imageIdFromUri(imageUri)),
    image_uri: imageUri,
    image_url: displayUrl(String(source.image_url || imageUri)),
    family: String(source.family || ""),
    zone_id: String(source.zone_id || ""),
    mold_id: source.mold_id ? String(source.mold_id) : null,
    session_id: source.session_id ? String(source.session_id) : null,
    created_at: source.created_at ? String(source.created_at) : undefined
  };
}

function uniqueImages(images: AnnotatableImage[]) {
  const seen = new Set<string>();
  return images.filter((image) => {
    const key = image.image_uri || image.id;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function normalizeLoadedBox(value: unknown): AnnotationBox | null {
  if (!value || typeof value !== "object") return null;
  const source = value as Record<string, unknown>;
  const bbox = source.bbox as number[] | undefined;
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;
  return {
    id: String(source.id || `box_${Math.random().toString(36).slice(2)}`),
    element_id: source.element_id ? String(source.element_id) : null,
    class_name: String(source.class_name || "piece"),
    bbox: normalizeBox(bbox as [number, number, number, number]),
    status: source.status === "missing" || source.status === "uncertain" ? source.status : "present",
    notes: source.notes ? String(source.notes) : null
  };
}

function normalizeBox(box: [number, number, number, number]): [number, number, number, number] {
  const x1 = clamp(Math.min(box[0], box[2]), 0, 1);
  const y1 = clamp(Math.min(box[1], box[3]), 0, 1);
  const x2 = clamp(Math.max(box[0], box[2]), 0, 1);
  const y2 = clamp(Math.max(box[1], box[3]), 0, 1);
  return [x1, y1, x2, y2];
}

function imageIdFromUri(imageUri: string) {
  if (imageUri.startsWith("local://")) return imageUri.replace("local://", "");
  const clean = imageUri.split("?")[0].split("#")[0];
  return (clean.split("/").pop() || clean || `img_${Date.now().toString(36)}`).replace(/[^a-zA-Z0-9_]+/g, "_");
}

async function loadZones(): Promise<Zone[]> {
  const supabaseZones = await selectSupabase("zones");
  const normalized = supabaseZones.map((item) => ({
    id: String(item.id || item.zone_id || fieldValue(item, "zone_id") || ""),
    family: String(item.family || fieldValue(item, "family") || ""),
    name: String(item.name || item.label || fieldValue(item, "name") || fieldValue(item, "label") || "")
  })).filter((zone) => zone.id);
  if (normalized.length) return normalized;
  const backendZones = await getJson("/v1/zones").catch(() => []);
  return Array.isArray(backendZones) && backendZones.length ? backendZones.map((item) => ({ id: item.id, family: fieldValue(item, "family"), name: fieldValue(item, "name") || item.id })) : DEFAULT_ZONES;
}

async function loadRecords(table: string, endpoint?: string): Promise<ResourceRecord[]> {
  const [dbRecords, backendRecords] = await Promise.all([
    selectSupabase(table),
    endpoint ? getJson(endpoint).catch(() => []) : Promise.resolve([])
  ]);
  return dedupeRecords([
    ...dbRecords,
    ...(Array.isArray(backendRecords) ? backendRecords as ResourceRecord[] : [])
  ]);
}

function dedupeRecords(records: ResourceRecord[]) {
  const seen = new Set<string>();
  return records
    .filter((record) => {
      const source = recordSource(record);
      const key = String(source.id || record.id || `${source.family || ""}:${source.zone_id || ""}:${source.created_at || record.created_at || ""}`);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .sort((left, right) => Date.parse(String(recordSource(right).created_at || right.created_at || "")) - Date.parse(String(recordSource(left).created_at || left.created_at || "")));
}

async function selectSupabase(table: string): Promise<ResourceRecord[]> {
  if (!SUPABASE_ENABLED) return [];
  const { data, error } = await Promise.race([
    supabase.from(table).select("*").limit(100).order("created_at", { ascending: false }),
    timeoutValue<{ data: ResourceRecord[]; error: null }>({ data: [], error: null }, 1200)
  ]);
  if (error) return [];
  return (data || []) as ResourceRecord[];
}

async function insertSupabase(table: string, payload: Record<string, unknown>) {
  if (!SUPABASE_ENABLED) return;
  try {
    await Promise.race([
      supabase.from(table).upsert(payload).select().maybeSingle(),
      timeoutValue(null, 1200)
    ]);
  } catch {
    return;
  }
}

function timeoutValue<T>(value: T, timeoutMs: number) {
  return new Promise<T>((resolve) => window.setTimeout(() => resolve(value), timeoutMs));
}

function maskPoints(mask: MaskRect) {
  return [
    { x: mask.left / 100, y: mask.top / 100 },
    { x: mask.right / 100, y: mask.top / 100 },
    { x: mask.right / 100, y: mask.bottom / 100 },
    { x: mask.left / 100, y: mask.bottom / 100 }
  ];
}

function stopCamera(streamRef: { current: MediaStream | null }) {
  streamRef.current?.getTracks().forEach((track) => track.stop());
  streamRef.current = null;
}

function cameraErrorMessage(error: unknown) {
  const text = messageFrom(error);
  if (/not found|device not found|notfound/i.test(text)) return "No se encontró una cámara disponible.";
  if (/denied|permission|notallowed/i.test(text)) return "Permiso de cámara bloqueado.";
  if (/constraint|overconstrained|notreadable/i.test(text)) return "No se pudo abrir la cámara.";
  return text;
}

function makeId(prefix: string, family: string, zoneId: string) {
  return `${prefix}_${family}_${zoneId}_${Date.now().toString(36)}`.replace(/[^a-zA-Z0-9_]+/g, "_").toLowerCase();
}

function fieldValue(record: ResourceRecord, key: string) {
  return String((record.data?.[key] ?? record[key] ?? "") as string);
}

function recordSource(record: ResourceRecord) {
  return (record.data && Object.keys(record.data).length ? record.data : record) as Record<string, unknown>;
}

function recordString(record: ResourceRecord, key: string) {
  const source = recordSource(record);
  return String(source[key] ?? "");
}

function publicRecord(record: ResourceRecord, hiddenKeys: string[]) {
  const source = { ...recordSource(record) } as Record<string, unknown>;
  for (const key of hiddenKeys) delete source[key];
  return source;
}

function toPlainRecord(record: InspectionResult) {
  return { id: record.id, status: record.status, message: record.message, result: record.result, evidence: record.evidence };
}

function statusLabel(status: InspectionStatus) {
  return { idle: "Esperando", uploading: "Subiendo", running: "Validando", correct: "Correcto", review: "Revisar", retake_photo: "Tomar otra foto", error: "Error" }[status];
}

function validationStatusLabel(status: MoldValidationSession["status"]) {
  return { pending: "Pendiente", in_progress: "En curso", complete: "Completo" }[status];
}

function syncStateLabel(status: "local" | "saving" | "saved" | "error") {
  return { local: "Local", saving: "Guardando", saved: "Servidor", error: "Local" }[status];
}

function formatNumber(value: number | null | undefined) {
  return typeof value === "number" ? value.toFixed(4) : "-";
}

function formatPercent(value: number | null | undefined) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "-";
}

function formatUnknown(value: unknown) {
  return typeof value === "number" ? value.toFixed(3) : value ? String(value) : "-";
}

function formatPercentValue(value: unknown) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : value ? String(value) : "-";
}

function inspectionSources(inspections: ResourceRecord[]): NormalizedInspection[] {
  return inspections.map((inspection) => {
    const source = recordSource(inspection);
    const result = source.result as InspectionResult["result"] | undefined;
    const rawStatus = String(source.status || "review");
    const status = rawStatus === "correct" || rawStatus === "retake_photo" || rawStatus === "review" ? rawStatus : "review";
    const family = String(source.family || result?.model_id || DEFAULT_ZONES[0].family).split("/")[0].trim() || DEFAULT_ZONES[0].family;
    const missingCount = Number(result?.piece_inspection?.missing_count || source.missing_count || 0);
    return {
      id: String(source.id || inspection.id || `inspection_${Date.now().toString(36)}`),
      family,
      zoneId: String(source.zone_id || result?.model_version || DEFAULT_ZONES[0].id),
      status,
      createdAt: String(source.created_at || inspection.created_at || ""),
      message: source.message ? String(source.message) : "",
      imageUri: source.image_uri ? String(source.image_uri) : undefined,
      missingCount: Number.isFinite(missingCount) ? missingCount : 0
    };
  });
}

function readingStats(readings: NormalizedInspection[]) {
  const correct = readings.filter((reading) => reading.status === "correct").length;
  const review = readings.filter((reading) => reading.status === "review").length;
  const retake = readings.filter((reading) => reading.status === "retake_photo").length;
  const total = readings.length;
  return {
    total,
    correct,
    review,
    retake,
    errorRate: total ? review / total : 0,
    retakeRate: total ? retake / total : 0,
    missingPieces: readings.reduce((sum, reading) => sum + reading.missingCount, 0)
  };
}

function buildLeadershipMetrics(
  molds: MoldSummary[],
  candidates: ResourceRecord[],
  jobs: ResourceRecord[],
  inspections: ResourceRecord[]
): LeadershipMetrics {
  const normalized = inspectionSources(inspections);
  const correctCount = normalized.filter((inspection) => inspection.status === "correct").length;
  const reviewCount = normalized.filter((inspection) => inspection.status === "review").length;
  const retakeCount = normalized.filter((inspection) => inspection.status === "retake_photo").length;
  const latestTraining = jobs
    .map((job) => String(recordSource(job).updated_at || recordSource(job).created_at || job.updated_at || job.created_at || ""))
    .filter(Boolean)
    .sort((left, right) => Date.parse(right) - Date.parse(left))[0];
  const promotedMetrics = candidateMetrics(candidates.find((candidate) => Boolean(recordSource(candidate).promoted)) || candidates[0]);
  return {
    totalValidations: normalized.length,
    correctCount,
    reviewCount,
    retakeCount,
    retakeRate: normalized.length ? retakeCount / normalized.length : 0,
    activeMolds: molds.length,
    missingPieces: normalized.reduce((total, inspection) => total + inspection.missingCount, 0),
    latestTraining,
    falsePassRate: promotedMetrics.falsePassRate,
    validationRecall: promotedMetrics.validationRecall
  };
}

function buildMoldValidationSummaries(
  molds: MoldSummary[],
  candidates: ResourceRecord[],
  inspections: ResourceRecord[]
): MoldValidationSummary[] {
  const rows = new Map<string, MoldValidationSummary>();
  for (const mold of molds) {
    rows.set(mold.id, {
      id: mold.id,
      name: mold.name,
      family: mold.family,
      zoneId: mold.zoneId,
      status: mold.status,
      validations: 0,
      correct: 0,
      review: 0,
      retake: 0,
      retakeRate: 0,
      missingPieces: 0,
      lastInspection: mold.lastInspection,
      lastTraining: mold.lastTraining,
      confidence: mold.confidence,
      falsePassRate: mold.falsePassRate,
      validationRecall: null
    });
  }

  for (const candidate of candidates) {
    const source = recordSource(candidate);
    const family = String(source.family || "");
    if (!family) continue;
    const zoneId = String(source.zone_id || DEFAULT_ZONES[0].id);
    const id = `${family}:${zoneId}`;
    const row = rows.get(id);
    if (!row) continue;
    const metrics = candidateMetrics(candidate);
    rows.set(id, {
      ...row,
      status: source.promoted ? "ready" : row.status,
      confidence: metrics.confidence ?? row.confidence,
      falsePassRate: metrics.falsePassRate ?? row.falsePassRate,
      validationRecall: metrics.validationRecall ?? row.validationRecall
    });
  }

  for (const inspection of inspectionSources(inspections)) {
    const id = `${inspection.family}:${inspection.zoneId}`;
    const current = rows.get(id) || {
      id,
      name: titleFromSlug(inspection.family),
      family: inspection.family,
      zoneId: inspection.zoneId,
      status: "needs_data" as MoldSummary["status"],
      validations: 0,
      correct: 0,
      review: 0,
      retake: 0,
      retakeRate: 0,
      missingPieces: 0,
      confidence: null,
      falsePassRate: null,
      validationRecall: null
    };
    const isLatest = !current.lastInspection || Date.parse(inspection.createdAt || "") >= Date.parse(current.lastInspection || "");
    const validations = current.validations + 1;
    const correct = current.correct + (inspection.status === "correct" ? 1 : 0);
    const review = current.review + (inspection.status === "review" ? 1 : 0);
    const retake = current.retake + (inspection.status === "retake_photo" ? 1 : 0);
    rows.set(id, {
      ...current,
      validations,
      correct,
      review,
      retake,
      retakeRate: validations ? retake / validations : 0,
      missingPieces: current.missingPieces + inspection.missingCount,
      lastInspection: isLatest ? inspection.createdAt : current.lastInspection,
      latestImageUri: isLatest ? inspection.imageUri : current.latestImageUri,
      latestMessage: isLatest ? inspection.message : current.latestMessage
    });
  }

  return Array.from(rows.values()).sort((left, right) => Date.parse(right.lastInspection || "") - Date.parse(left.lastInspection || ""));
}

function buildInspectionTrend(inspections: ResourceRecord[]): InspectionTrendPoint[] {
  const points = new Map<string, InspectionTrendPoint>();
  for (let offset = 6; offset >= 0; offset -= 1) {
    const date = new Date();
    date.setHours(0, 0, 0, 0);
    date.setDate(date.getDate() - offset);
    const key = localDateKey(date);
    points.set(key, { date: key, total: 0, correct: 0, review: 0, retake: 0 });
  }
  for (const inspection of inspectionSources(inspections)) {
    if (!inspection.createdAt || !withinDays(inspection.createdAt, 7)) continue;
    const key = localDateKey(new Date(inspection.createdAt));
    const point = points.get(key);
    if (!point) continue;
    point.total += 1;
    if (inspection.status === "correct") point.correct += 1;
    if (inspection.status === "review") point.review += 1;
    if (inspection.status === "retake_photo") point.retake += 1;
  }
  return Array.from(points.values());
}

function candidateMetrics(candidate?: ResourceRecord) {
  const metrics = candidate ? (recordSource(candidate).metrics || {}) as Record<string, unknown> : {};
  return {
    confidence: typeof metrics.confidence === "number" ? metrics.confidence : null,
    falsePassRate: typeof metrics.false_pass_rate === "number" ? metrics.false_pass_rate : null,
    validationRecall: typeof metrics.validation_recall === "number" ? metrics.validation_recall : null
  };
}

function withinDays(value: string | undefined, days: number) {
  if (!value) return true;
  const time = Date.parse(value);
  if (Number.isNaN(time)) return true;
  return Date.now() - time <= days * 24 * 60 * 60 * 1000;
}

function shortDay(value: string) {
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("es-MX", { weekday: "short" }).format(date).replace(".", "");
}

function localDateKey(date: Date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function buildMoldSummaries(
  recipes: ResourceRecord[],
  datasets: ResourceRecord[],
  candidates: ResourceRecord[],
  jobs: ResourceRecord[],
  inspections: ResourceRecord[]
): MoldSummary[] {
  const keys = new Map<string, Partial<MoldSummary>>();

  function ensure(family: string, zoneId: string, name?: string) {
    const id = `${family}:${zoneId}`;
    const current = keys.get(id) || { id, family, zoneId, name: name || titleFromSlug(family) };
    keys.set(id, { ...current, name: name || current.name || titleFromSlug(family) });
    return id;
  }

  function touch(id: string, createdAt?: string, actionAt?: string) {
    const current = keys.get(id) || {};
    const nextCreated = earliestDate(current.createdAt, createdAt);
    const nextAction = latestDate(current.lastActionAt, actionAt || createdAt);
    keys.set(id, { ...current, createdAt: nextCreated, lastActionAt: nextAction });
  }

  for (const recipe of recipes) {
    const family = recordString(recipe, "family") || fieldValue(recipe, "family") || String(recipe.id || "molde_demo");
    const zoneId = recordString(recipe, "zone_id") || DEFAULT_ZONES[0].id;
    const id = ensure(family, zoneId, recordString(recipe, "name") || String(recipe.id));
    touch(id, recordDate(recipe, "created_at"), recordDate(recipe, "updated_at"));
  }

  for (const dataset of datasets) {
    const family = recordString(dataset, "family") || fieldValue(dataset, "family");
    const zoneId = recordString(dataset, "zone_id") || fieldValue(dataset, "zone_id") || DEFAULT_ZONES[0].id;
    if (!family) continue;
    const id = ensure(family, zoneId);
    touch(id, recordDate(dataset, "created_at"), recordDate(dataset, "updated_at"));
    const current = keys.get(id) || {};
    keys.set(id, {
      ...current,
      datasetId: String(dataset.id),
      okCount: Number(recordString(dataset, "ok_count") || fieldValue(dataset, "ok_count") || 0),
      faultCount: Number(recordString(dataset, "fault_count") || fieldValue(dataset, "fault_count") || 0),
      pieceCount: Number(recordString(dataset, "piece_count") || fieldValue(dataset, "piece_count") || 0)
    });
  }

  for (const candidate of candidates) {
    const source = recordSource(candidate);
    const family = String(source.family || "");
    const zoneId = String(source.zone_id || DEFAULT_ZONES[0].id);
    if (!family) continue;
    const id = ensure(family, zoneId);
    touch(id, recordDate(candidate, "created_at"), recordDate(candidate, "updated_at"));
    const current = keys.get(id) || {};
    const metrics = (source.metrics || {}) as Record<string, unknown>;
    const confidence = typeof metrics.confidence === "number" ? metrics.confidence : current.confidence;
    const falsePassRate = typeof metrics.false_pass_rate === "number" ? metrics.false_pass_rate : current.falsePassRate;
    keys.set(id, {
      ...current,
      confidence: confidence ?? null,
      falsePassRate: falsePassRate ?? null,
      status: source.promoted ? "ready" : current.status
    });
  }

  for (const job of jobs) {
    const source = recordSource(job);
    const family = String(source.family || "");
    const zoneId = String(source.zone_id || DEFAULT_ZONES[0].id);
    if (!family) continue;
    const id = ensure(family, zoneId);
    const jobStatus = String(source.status || "").toLowerCase();
    const actionAt = String(source.updated_at || source.created_at || job.updated_at || job.created_at || "");
    touch(id, recordDate(job, "created_at"), actionAt);
    const current = keys.get(id) || {};
    keys.set(id, {
      ...current,
      lastTraining: actionAt,
      status: jobStatus.includes("running") || jobStatus.includes("queued") ? "training" : current.status
    });
  }

  for (const inspection of inspections) {
    const source = recordSource(inspection);
    const result = source.result as InspectionResult["result"] | undefined;
    const family = String(source.family || result?.model_id || "").split("/")[0].trim();
    const zoneId = String(source.zone_id || DEFAULT_ZONES[0].id);
    if (!family) continue;
    const id = ensure(family, zoneId);
    const createdAt = String(source.created_at || inspection.created_at || "");
    touch(id, createdAt, createdAt);
    const current = keys.get(id) || {};
    keys.set(id, { ...current, lastInspection: createdAt });
  }

  const summaries = Array.from(keys.values()).map((item) => {
    const okCount = item.okCount || 0;
    const faultCount = item.faultCount || 0;
    const hasDataset = Boolean(item.datasetId || okCount || faultCount);
    const status = item.status || (!hasDataset ? "needs_data" : "no_model");
    return {
      id: item.id || `${item.family}:${item.zoneId}`,
      name: item.name || titleFromSlug(item.family || "molde"),
      family: item.family || "molde_demo",
      zoneId: item.zoneId || DEFAULT_ZONES[0].id,
      status,
      datasetId: item.datasetId,
      okCount,
      faultCount,
      pieceCount: item.pieceCount || 0,
      confidence: item.confidence ?? null,
      falsePassRate: item.falsePassRate ?? null,
      createdAt: item.createdAt,
      lastActionAt: item.lastActionAt,
      lastTraining: item.lastTraining,
      lastInspection: item.lastInspection
    } as MoldSummary;
  });

  return summaries.length ? summaries : [{
    id: "molde_demo:frontal_zona_01",
    name: "Molde demo",
    family: "molde_demo",
    zoneId: "frontal_zona_01",
    status: "needs_data",
    okCount: 0,
    faultCount: 0,
    pieceCount: 0,
    confidence: null,
    falsePassRate: null
  }];
}

function dateRank(value?: string) {
  const time = value ? Date.parse(value) : 0;
  return Number.isFinite(time) ? time : 0;
}

function latestDate(left?: string, right?: string) {
  if (!left) return right;
  if (!right) return left;
  return dateRank(right) > dateRank(left) ? right : left;
}

function earliestDate(left?: string, right?: string) {
  if (!left) return right;
  if (!right) return left;
  return dateRank(right) < dateRank(left) ? right : left;
}

function recordDate(record: ResourceRecord, key: string) {
  const source = recordSource(record);
  return String(source[key] || record[key] || "");
}

function ProgressSteps({ mold }: { mold: MoldSummary }) {
  const steps = [
    { label: "Datos", done: Boolean(mold.datasetId || mold.okCount || mold.faultCount) },
    { label: "Entrena", done: Boolean(mold.lastTraining || mold.status === "ready") },
    { label: "IA lista", done: mold.status === "ready" },
    { label: "Prueba", done: Boolean(mold.lastInspection) }
  ];
  return <div className="progressSteps">{steps.map((step) => <span key={step.label} className={step.done ? "done" : ""}>{step.label}</span>)}</div>;
}

function QualityBars({ mold }: { mold: MoldSummary }) {
  const confidence = typeof mold.confidence === "number" ? mold.confidence : 0;
  const falsePassSafety = typeof mold.falsePassRate === "number" ? Math.max(0, 1 - mold.falsePassRate) : 0;
  const dataBalance = mold.okCount && mold.faultCount ? Math.min(mold.okCount, mold.faultCount) / Math.max(mold.okCount, mold.faultCount) : 0;
  return (
    <div className="qualityBars">
      <QualityBar label="Confianza" value={confidence} />
      <QualityBar label="Evita falsos aprobados" value={falsePassSafety} />
      <QualityBar label="Balance de datos" value={dataBalance} />
    </div>
  );
}

function QualityBar({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <span>{label}</span>
      <i><b style={{ width: `${Math.round(value * 100)}%` }} /></i>
      <strong>{value ? `${Math.round(value * 100)}%` : "Pendiente"}</strong>
    </div>
  );
}

function slugFromName(name: string) {
  const normalized = name.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  return normalized.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || `molde_${Date.now().toString(36)}`;
}

function titleFromSlug(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function moldStatusLabel(status: MoldSummary["status"]) {
  return {
    ready: "Listo",
    training: "Entrenando",
    needs_data: "Requiere datos",
    no_model: "Sin modelo"
  }[status];
}

function statusClassForMold(status: MoldSummary["status"]) {
  return status === "ready" ? "correct" : status === "training" ? "running" : status === "no_model" ? "review" : "retake_photo";
}

function formatRisk(value: number | null) {
  if (typeof value !== "number") return "Pendiente";
  if (value <= 0.01) return "Bajo";
  if (value <= 0.05) return "Medio";
  return "Alto";
}

function formatDate(value?: string) {
  if (!value) return "Sin fecha";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Sin fecha";
  return new Intl.DateTimeFormat("es-MX", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function loadSectionPlan(family: string, moldKey: string, seedZoneId: string): MoldSectionPlan {
  try {
    const raw = window.localStorage.getItem(sectionPlanStorageKey(family, moldKey));
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<MoldSectionPlan>;
      if (parsed.family && Array.isArray(parsed.sections) && parsed.sections.length) {
        return {
          moldKey: parsed.moldKey || moldKey || family,
          family: parsed.family,
          sections: parsed.sections.map(normalizeSection).filter(Boolean) as MoldSection[],
          updatedAt: parsed.updatedAt || new Date().toISOString()
        };
      }
    }
  } catch {
    // fall through to default section plan
  }
  return {
    moldKey: moldKey || family,
    family,
    sections: [{
      id: "section_01_front",
      zoneId: seedZoneId || DEFAULT_ZONES[0].id,
      label: "Zona 1 / frente",
      zoneIndex: 1,
      view: "front"
    }],
    updatedAt: new Date().toISOString()
  };
}

function persistSectionPlan(plan: MoldSectionPlan) {
  window.localStorage.setItem(sectionPlanStorageKey(plan.family, plan.moldKey), JSON.stringify(plan));
}

function buildSectionPlan(family: string, moldKey: string, zoneCount: number, views: MoldViewSide[]): MoldSectionPlan {
  const sections: MoldSection[] = [];
  for (let zoneIndex = 1; zoneIndex <= zoneCount; zoneIndex += 1) {
    for (const view of views) {
      const zoneToken = `zona_${String(zoneIndex).padStart(2, "0")}_${view}`;
      sections.push({
        id: `section_${String(zoneIndex).padStart(2, "0")}_${view}`,
        zoneId: zoneToken,
        label: `Zona ${zoneIndex} / ${viewLabel(view).toLowerCase()}`,
        zoneIndex,
        view
      });
    }
  }
  return { moldKey: moldKey || family, family, sections, updatedAt: new Date().toISOString() };
}

function selectedViewsFromPlan(plan: MoldSectionPlan): MoldViewSide[] {
  const views = VIEW_OPTIONS.map((option) => option.value).filter((view) => plan.sections.some((section) => section.view === view));
  return views.length ? views : ["front"];
}

function normalizeSection(value: unknown): MoldSection | null {
  if (!value || typeof value !== "object") return null;
  const source = value as Record<string, unknown>;
  const view = source.view === "left" || source.view === "right" || source.view === "front" ? source.view : "front";
  const zoneIndex = Number(source.zoneIndex || source.zone_index || 1);
  const zoneId = String(source.zoneId || source.zone_id || `zona_${String(zoneIndex).padStart(2, "0")}_${view}`);
  return {
    id: String(source.id || `section_${String(zoneIndex).padStart(2, "0")}_${view}`),
    zoneId,
    label: String(source.label || `Zona ${zoneIndex} / ${viewLabel(view).toLowerCase()}`),
    zoneIndex: Number.isFinite(zoneIndex) ? zoneIndex : 1,
    view
  };
}

function viewLabel(view: MoldViewSide) {
  return VIEW_OPTIONS.find((option) => option.value === view)?.label || view;
}

function sectionPlanStorageKey(family: string, moldKey: string) {
  return `mold_vision_section_plan:${family}:${moldKey || family}`;
}

function loadMoldDrafts(): MoldDrafts {
  try {
    const raw = window.localStorage.getItem("mold_vision_mold_drafts");
    if (!raw) return { names: {}, deleted: [] };
    const parsed = JSON.parse(raw) as Partial<MoldDrafts>;
    return {
      names: parsed.names && typeof parsed.names === "object" ? parsed.names : {},
      deleted: Array.isArray(parsed.deleted) ? parsed.deleted : []
    };
  } catch {
    return { names: {}, deleted: [] };
  }
}

function saveMoldDrafts(drafts: MoldDrafts) {
  window.localStorage.setItem("mold_vision_mold_drafts", JSON.stringify(drafts));
}

const rootElement = document.getElementById("root")! as HTMLElement & { _moldVisionRoot?: ReturnType<typeof createRoot> };
const root = rootElement._moldVisionRoot || createRoot(rootElement);
rootElement._moldVisionRoot = root;
root.render(<React.StrictMode><App /></React.StrictMode>);
