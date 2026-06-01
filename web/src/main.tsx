import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { supabase } from "./utils/supabase";
import "./styles.css";

type InspectionStatus = "idle" | "uploading" | "running" | "correct" | "review" | "retake_photo" | "error";
type View = "molds" | "segmenter" | "capture" | "history";

type Zone = {
  id: string;
  family: string;
  name?: string;
  label?: string;
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
  lastTraining?: string;
  lastInspection?: string;
};

type Point = { x: number; y: number };

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

const API_BASE = import.meta.env.VITE_API_BASE || "";
const DEFAULT_ZONES: Zone[] = [{ id: "frontal_zona_01", family: "molde_demo", name: "Frontal zona 01" }];
const DEFAULT_SEGMENTER_POLYGON: Point[] = [
  { x: 0.18, y: 0.22 },
  { x: 0.82, y: 0.18 },
  { x: 0.88, y: 0.72 },
  { x: 0.28, y: 0.82 },
  { x: 0.14, y: 0.56 }
];

function App() {
  const [view, setView] = useState<View>("molds");
  const [family, setFamily] = useState("molde_demo");
  const [zoneId, setZoneId] = useState("frontal_zona_01");
  const [moldId, setMoldId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [captureOpen, setCaptureOpen] = useState(false);
  const [status, setStatus] = useState<InspectionStatus>("idle");
  const [message, setMessage] = useState("Lista para capturar una zona.");
  const [result, setResult] = useState<InspectionResult | null>(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  async function runInspection(targetFile = file) {
    if (!targetFile) {
      setStatus("error");
      setMessage("Selecciona o toma una foto antes de validar.");
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
      setMessage("Validando captura y modelo...");
      const inspection = (await postJson("/v1/inspections", {
        family,
        zone_id: zoneId,
        mold_id: moldId || null,
        session_id: null,
        image_uri: presign.object_uri,
        capture_metadata: { source: "web" }
      })) as InspectionResult;
      setResult(inspection);
      setStatus(inspection.status);
      setMessage(inspection.message);
      await insertSupabase("inspections", toPlainRecord(inspection));
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
    setFamily(mold.family);
    setZoneId(mold.zoneId);
    setMoldId(mold.id);
    setFile(null);
    setResult(null);
    setStatus("idle");
    setMessage("Lista para capturar una zona.");
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="mark">MV</span>
          <div>
            <strong>Mold Vision</strong>
            <small>Inspección cloud</small>
          </div>
        </div>
        <nav>
          <button className={view === "molds" ? "active" : ""} onClick={() => setView("molds")}>Moldes</button>
          <button className={view === "segmenter" ? "active" : ""} onClick={() => setView("segmenter")}>Guía de cámara</button>
          <button className={view === "capture" ? "active" : ""} onClick={() => setView("capture")}>Captura</button>
          <button className={view === "history" ? "active" : ""} onClick={() => setView("history")}>Historial</button>
        </nav>
      </aside>

      <section className="workspace">
        {view === "molds" && <MoldsView onTest={(mold) => { selectMold(mold); setView("capture"); }} />}

        {view === "segmenter" && <SegmenterTrainingView />}

        {view === "capture" && (
          <>
            <header className="topbar">
              <div>
                <h1>Captura</h1>
                <p>Selecciona el molde y toma una foto guiada. El sistema valida la imagen antes de inspeccionar.</p>
              </div>
              <button className="primary" onClick={() => setCaptureOpen(true)} disabled={status === "uploading" || status === "running"}>Capturar foto</button>
            </header>

            <MoldQuickSelect family={family} zoneId={zoneId} onSelect={selectMold} />

            <section className="captureLayout">
              <div className="panel capture">
                <div className="captureHeader">
                  <strong>Foto para inspección</strong>
                  <span className={`status ${status}`}>{statusLabel(status)}</span>
                </div>
                <CaptureStillPanel previewUrl={previewUrl} status={status} onOpen={() => setCaptureOpen(true)} onFile={setFile} onInspect={runInspection} />
                <div className="hintRow">
                  <span>Objeto completo dentro del marco</span>
                  <span>Sin brillo extremo</span>
                  <span>Sin desenfoque</span>
                </div>
              </div>
            </section>

            {result ? <InspectionResultPanel status={status} message={message} result={result} previewUrl={previewUrl} family={family} zoneId={zoneId} /> : null}
            {captureOpen ? <CameraCapture family={family} zoneId={zoneId} previewUrl={previewUrl} onFile={setFile} onCapture={handleCaptured} onCancel={() => setCaptureOpen(false)} fullscreen /> : null}
          </>
        )}

        {view === "history" && <HistoryView />}
      </section>
    </main>
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
  const [okFiles, setOkFiles] = useState<File[]>([]);
  const [faultFiles, setFaultFiles] = useState<File[]>([]);
  const [testFile, setTestFile] = useState<File | null>(null);
  const [testResult, setTestResult] = useState<InspectionResult | null>(null);
  const [drafts, setDrafts] = useState<MoldDrafts>(() => loadMoldDrafts());
  const [editingName, setEditingName] = useState("");
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const molds = useMemo(() => {
    return buildMoldSummaries(recipes, datasets, candidates, jobs, inspections)
      .filter((mold) => !drafts.deleted.includes(mold.id))
      .map((mold) => ({ ...mold, name: drafts.names[mold.id] || mold.name }));
  }, [recipes, datasets, candidates, jobs, inspections, drafts]);
  const selectedMold = molds.find((mold) => mold.id === selectedId) || molds[0] || null;
  const testPreviewUrl = useMemo(() => (testFile ? URL.createObjectURL(testFile) : ""), [testFile]);

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (!selectedId && molds[0]) setSelectedId(molds[0].id);
  }, [molds, selectedId]);

  useEffect(() => {
    if (selectedMold && !editing) setEditingName(selectedMold.name);
  }, [selectedMold, editing]);

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

  async function registerMold() {
    const name = newMoldName.trim();
    if (!name) {
      setError("Escribe un nombre para el molde.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const family = slugFromName(name);
      const moldId = `${family}:${DEFAULT_ZONES[0].id}`;
      const recipe = await postJson("/v1/recipes", {
        family,
        zone_id: DEFAULT_ZONES[0].id,
        name,
        objective: "presence_absence"
      });
      await insertSupabase("recipes", recipe as Record<string, unknown>);
      saveMoldDrafts({ names: { ...drafts.names, [moldId]: name }, deleted: drafts.deleted.filter((id) => id !== moldId) });
      setDrafts(loadMoldDrafts());
      setNewMoldName("");
      await refresh();
      setSelectedId(moldId);
    } catch (saveError) {
      setError(messageFrom(saveError));
      setLoading(false);
    }
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
    if (recipe?.id) {
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
    if (recipe?.id) {
      await supabase.from("recipes").delete().eq("id", recipe.id);
    }
  }

  async function saveDataset(mold: MoldSummary) {
    if (!okFiles.length || !faultFiles.length) {
      setError("Sube fotos correctas y fotos con pieza faltante antes de guardar el dataset.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const [okImageUris, faultImageUris] = await Promise.all([
        uploadFiles(okFiles, mold.family, mold.zoneId, "dataset"),
        uploadFiles(faultFiles, mold.family, mold.zoneId, "dataset")
      ]);
      const dataset = await postJson("/v1/datasets/from-examples", {
        name: `Dataset ${mold.name}`,
        family: mold.family,
        zone_id: mold.zoneId,
        ok_image_uris: okImageUris,
        fault_image_uris: faultImageUris,
        mask: { type: "auto" }
      });
      await insertSupabase("datasets", dataset as Record<string, unknown>);
      setOkFiles([]);
      setFaultFiles([]);
      await refresh();
    } catch (saveError) {
      setError(messageFrom(saveError));
      setLoading(false);
    }
  }

  async function generateModel(mold: MoldSummary) {
    if (!mold.datasetId) {
      setError("Primero sube el dataset del molde.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const job = await postJson("/v1/inspector-training-jobs", {
        family: mold.family,
        zone_id: mold.zoneId,
        dataset_id: mold.datasetId,
        target: "cloud-gpu"
      });
      await insertSupabase("inspector_training_jobs", job as Record<string, unknown>);
      await refresh();
    } catch (trainingError) {
      setError(messageFrom(trainingError));
      setLoading(false);
    }
  }

  async function testMold(mold: MoldSummary) {
    if (!testFile) {
      setError("Sube una foto para probar el modelo.");
      return;
    }
    setLoading(true);
    setError("");
    setTestResult(null);
    try {
      const imageUri = await uploadSystemFile(testFile, mold.family, mold.zoneId, "inspection");
      const inspection = (await postJson("/v1/inspections", {
        family: mold.family,
        zone_id: mold.zoneId,
        mold_id: mold.id,
        session_id: null,
        image_uri: imageUri,
        capture_metadata: { source: "mold_test" }
      })) as InspectionResult;
      await insertSupabase("inspections", toPlainRecord(inspection));
      setTestResult(inspection);
      await refresh();
    } catch (inspectionError) {
      setError(messageFrom(inspectionError));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <header className="topbar">
        <div>
          <h1>Moldes</h1>
          <p>Administra datasets, entrenamiento y pruebas desde un solo lugar.</p>
        </div>
        <button className="primary" onClick={refresh} disabled={loading}>Actualizar</button>
      </header>

      <section className="moldRegister">
        <div>
          <strong>Registrar nuevo molde</strong>
          <span>Asigna un nombre, carga ejemplos y genera el modelo de IA.</span>
        </div>
        <input value={newMoldName} onChange={(event) => setNewMoldName(event.target.value)} placeholder="Nombre del molde" />
        <button className="primary" onClick={registerMold} disabled={loading}>Registrar nuevo molde</button>
      </section>

      {error ? <p className="errorText inlineError">{error}</p> : null}

      <section className="moldsWorkspace">
        <aside className="moldList">
          {molds.length ? molds.map((mold) => (
            <button key={mold.id} className={selectedMold?.id === mold.id ? "moldRow active" : "moldRow"} onClick={() => setSelectedId(mold.id)}>
              <span>
                <strong>{mold.name}</strong>
                <small>{mold.lastInspection ? `Última inspección ${formatDate(mold.lastInspection)}` : "Sin inspecciones"}</small>
              </span>
              <i className={`moldState ${mold.status}`}>{moldStatusLabel(mold.status)}</i>
            </button>
          )) : (
            <div className="emptyMolds">
              <strong>Sin moldes registrados</strong>
              <span>Registra el primer molde para comenzar.</span>
            </div>
          )}
        </aside>

        <section className="moldDetail">
          {selectedMold ? (
            <>
              <div className="moldHero">
                <div>
                  <span className={`status ${statusClassForMold(selectedMold.status)}`}>{moldStatusLabel(selectedMold.status)}</span>
                  {editing ? (
                    <div className="editMoldName">
                      <input value={editingName} onChange={(event) => setEditingName(event.target.value)} aria-label="Nombre del molde" />
                      <button className="primary" onClick={() => updateMoldName(selectedMold)} disabled={loading}>Guardar</button>
                      <button className="secondary" onClick={() => setEditing(false)}>Cancelar</button>
                    </div>
                  ) : <h2>{selectedMold.name}</h2>}
                  <p>{selectedMold.lastTraining ? `Último entrenamiento ${formatDate(selectedMold.lastTraining)}` : "Sin entrenamiento generado todavía."}</p>
                </div>
                <div className="heroActions">
                  <button className="secondary" onClick={() => { setEditingName(selectedMold.name); setEditing(true); }}>Editar</button>
                  <button className="dangerButton" onClick={() => deleteMold(selectedMold)}>Borrar</button>
                  <button className="secondary" onClick={() => onTest(selectedMold)}>Capturar foto</button>
                </div>
              </div>

              <div className="userMetrics">
                <Metric label="Fotos de entrenamiento" value={String(selectedMold.okCount + selectedMold.faultCount)} />
                <Metric label="Confianza promedio" value={formatPercent(selectedMold.confidence)} />
                <Metric label="Riesgo falso aprobado" value={formatRisk(selectedMold.falsePassRate)} />
                <Metric label="Piezas verificadas" value={selectedMold.pieceCount ? String(selectedMold.pieceCount) : "Pendiente"} />
              </div>

              <div className="chartsGrid">
                <section className="panel chartPanel">
                  <strong>Progreso del modelo</strong>
                  <ProgressSteps mold={selectedMold} />
                </section>
                <section className="panel chartPanel">
                  <strong>Calidad entendible</strong>
                  <QualityBars mold={selectedMold} />
                </section>
              </div>

              <div className="moldActionGrid">
                <section className="panel uploadPanel">
                  <div className="captureHeader">
                    <strong>Dataset del molde</strong>
                    <span>{selectedMold.okCount} correctas / {selectedMold.faultCount} faltantes</span>
                  </div>
                  <div className="uploadColumns">
                    <FilePicker label="Fotos correctas" detail="Molde completo" count={okFiles.length} multiple onChange={setOkFiles} />
                    <FilePicker label="Fotos con pieza faltante" detail="Molde incompleto" count={faultFiles.length} multiple onChange={setFaultFiles} />
                  </div>
                  <button className="primary fullWidth" onClick={() => saveDataset(selectedMold)} disabled={loading}>Guardar dataset</button>
                </section>

                <section className="panel uploadPanel">
                  <div className="captureHeader">
                    <strong>Modelo de IA</strong>
                    <span>{selectedMold.datasetId ? "Dataset listo" : "Falta dataset"}</span>
                  </div>
                  <p className="fieldNote">El sistema entrena candidatos y conserva automáticamente el de mejor desempeño.</p>
                  <button className="primary fullWidth" onClick={() => generateModel(selectedMold)} disabled={loading}>Generar modelo de IA</button>
                </section>
              </div>

              <section className="panel testPanel">
                <div className="captureHeader">
                  <strong>Probar modelo</strong>
                  <span>Sube una foto y revisa el resultado</span>
                </div>
                <div className="testGrid">
                  <div className="testControls">
                    <FilePicker label="Subir foto de prueba" detail="Imagen individual" count={testFile ? 1 : 0} onChange={(files) => setTestFile(files[0] ?? null)} />
                    <button className="primary" onClick={() => testMold(selectedMold)} disabled={loading || !testFile}>Revisar resultado</button>
                  </div>
                  {testResult ? (
                    <>
                      <EvaluationImage imageUrl={testPreviewUrl || inspectionImageUrl(testResult)} missingPolygons={missingPiecePolygons(testResult)} />
                      <div className="resultCopy">
                        <span className={`resultBadge ${testResult.status}`}>{statusLabel(testResult.status)}</span>
                        <strong>{testResult.message}</strong>
                        <p>{missingPiecePolygons(testResult).length ? "Se marcaron regiones faltantes en rojo." : "Sin regiones faltantes detectadas."}</p>
                      </div>
                    </>
                  ) : (
                    <div className="pendingResult">
                      <strong>Sin análisis todavía</strong>
                      <p>El resultado aparecerá solo después de revisar una foto.</p>
                    </div>
                  )}
                </div>
              </section>
            </>
          ) : null}
        </section>
      </section>
    </>
  );
}

function SegmenterTrainingView() {
  const [name, setName] = useState("Detector de contorno del molde");
  const [files, setFiles] = useState<File[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [polygon, setPolygon] = useState<Point[]>(DEFAULT_SEGMENTER_POLYGON);
  const [draggingPoint, setDraggingPoint] = useState<number | null>(null);
  const [datasets, setDatasets] = useState<ResourceRecord[]>([]);
  const [jobs, setJobs] = useState<ResourceRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const editorRef = useRef<HTMLDivElement | null>(null);
  const selectedFile = files[selectedIndex] || files[0] || null;
  const previewUrl = useMemo(() => (selectedFile ? URL.createObjectURL(selectedFile) : ""), [selectedFile]);

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const [loadedDatasets, loadedJobs] = await Promise.all([
        loadRecords("segmenter_datasets", "/v1/segmenter_datasets"),
        loadRecords("segmenter_training_jobs", "/v1/segmenter_training_jobs")
      ]);
      setDatasets(loadedDatasets);
      setJobs(loadedJobs);
    } catch (loadError) {
      setError(messageFrom(loadError));
    } finally {
      setLoading(false);
    }
  }

  async function createDataset() {
    if (!files.length) {
      setError("Sube al menos una foto del molde con el contorno marcado.");
      return;
    }
    if (polygon.length < 3) {
      setError("Marca al menos tres puntos para formar el polígono del molde.");
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
          polygon
        }))
      });
      await insertSupabase("segmenter_datasets", dataset as Record<string, unknown>);
      setFiles([]);
      await refresh();
    } catch (saveError) {
      setError(messageFrom(saveError));
      setLoading(false);
    }
  }

  function pointFromPointer(event: React.PointerEvent<HTMLElement>) {
    const rect = editorRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0.5, y: 0.5 };
    return {
      x: clamp((event.clientX - rect.left) / rect.width, 0, 1),
      y: clamp((event.clientY - rect.top) / rect.height, 0, 1)
    };
  }

  function addPolygonPoint(event: React.PointerEvent<HTMLDivElement>) {
    if (draggingPoint !== null) return;
    const point = pointFromPointer(event);
    setPolygon((current) => insertPointInClosestSegment(current, point));
  }

  function movePolygonPoint(event: React.PointerEvent<HTMLDivElement>) {
    if (draggingPoint === null) return;
    const point = pointFromPointer(event);
    setPolygon((current) => current.map((existing, index) => index === draggingPoint ? point : existing));
  }

  function removeLastPoint() {
    setPolygon((current) => current.length > 3 ? current.slice(0, -1) : current);
  }

  async function trainLatest() {
    const dataset = datasets[0];
    if (!dataset) {
      setError("Primero guarda fotos con el contorno del molde.");
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
          <h1>Guía de cámara</h1>
          <p>Entrena la red que detecta el contorno del molde y guía al usuario al tomar la foto.</p>
        </div>
        <button className="primary" onClick={refresh} disabled={loading}>Actualizar</button>
      </header>

      {error ? <p className="errorText inlineError">{error}</p> : null}

      <section className="segmenterWorkspace">
        <aside className="panel segmenterSetup">
          <label>Nombre del entrenamiento
            <input value={name} onChange={(event) => setName(event.target.value)} />
          </label>
          <FilePicker label="Fotos del molde" detail="Marca el contorno en la vista previa" count={files.length} multiple onChange={setFiles} />
          {files.length ? (
            <label>Foto activa
              <select value={selectedIndex} onChange={(event) => setSelectedIndex(Number(event.target.value))}>
                {files.map((file, index) => <option key={`${file.name}-${index}`} value={index}>{file.name}</option>)}
              </select>
            </label>
          ) : null}
          <button className="primary fullWidth" onClick={createDataset} disabled={loading}>Guardar contornos</button>
          <button className="secondary fullWidth" onClick={trainLatest} disabled={loading}>Entrenar guía de cámara</button>
        </aside>

        <section className="panel polygonEditor">
          <div className="captureHeader">
            <strong>Contorno del molde</strong>
            <span>Haz clic para agregar puntos. Arrastra los vértices para ajustar.</span>
          </div>
          <div
            ref={editorRef}
            className="maskPreview polygonCanvas"
            onPointerDown={addPolygonPoint}
            onPointerMove={movePolygonPoint}
            onPointerUp={() => setDraggingPoint(null)}
            onPointerCancel={() => setDraggingPoint(null)}
            onPointerLeave={() => setDraggingPoint(null)}
          >
            {previewUrl ? <img src={previewUrl} alt="Molde anotado" /> : <span>Sube una foto y ajusta el contorno del molde</span>}
            <svg className="polygonOverlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              <polygon points={polygonPoints(polygon)} />
              <polyline points={`${polygonPoints(polygon)} ${polygon[0].x * 100},${polygon[0].y * 100}`} />
              {polygon.map((point, index) => (
                <circle
                  key={`${point.x}-${point.y}-${index}`}
                  cx={point.x * 100}
                  cy={point.y * 100}
                  r="1.9"
                  onPointerDown={(event) => {
                    event.stopPropagation();
                    setDraggingPoint(index);
                  }}
                />
              ))}
            </svg>
            <div className="frame"><i /><i /><i /><i /></div>
          </div>
          <div className="polygonToolbar">
            <span>{polygon.length} puntos</span>
            <button className="secondary" type="button" onClick={removeLastPoint} disabled={polygon.length <= 3}>Deshacer punto</button>
            <button className="secondary" type="button" onClick={() => setPolygon(DEFAULT_SEGMENTER_POLYGON)}>Reiniciar polígono</button>
          </div>
        </section>

        <aside className="panel trainingStatus">
          <strong>Estado</strong>
          <Metric label="Fotos con contorno" value={String(datasets.reduce((total, item) => total + Number(recordString(item, "image_count") || 0), 0))} />
          <Metric label="Entrenamientos" value={String(jobs.length)} />
          <Metric label="Último estado" value={jobs[0] ? titleFromSlug(recordString(jobs[0], "status") || "Registrado") : "Sin entrenamiento"} />
        </aside>
      </section>
    </>
  );
}

function MoldQuickSelect({ family, zoneId, onSelect }: { family: string; zoneId: string; onSelect: (mold: MoldSummary) => void }) {
  const [molds, setMolds] = useState<MoldSummary[]>([]);

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

  return (
    <section className="captureMoldSelect">
      <label>Molde
        <select value={`${family}:${zoneId}`} onChange={(event) => {
          const mold = molds.find((item) => item.id === event.target.value);
          if (mold) onSelect(mold);
        }}>
          {molds.length ? molds.map((mold) => <option key={mold.id} value={mold.id}>{mold.name}</option>) : <option value={`${family}:${zoneId}`}>Molde demo</option>}
        </select>
      </label>
    </section>
  );
}

function HistoryView() {
  const [records, setRecords] = useState<ResourceRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function refresh() {
      setLoading(true);
      setError("");
      try {
        setRecords(await loadRecords("inspections", "/v1/inspections"));
      } catch (loadError) {
        setError(messageFrom(loadError));
      } finally {
        setLoading(false);
      }
    }
    void refresh();
  }, []);

  return (
    <>
      <header className="topbar">
        <div>
          <h1>Historial</h1>
          <p>Últimas inspecciones y resultados operativos.</p>
        </div>
      </header>
      <section className="historyList">
        {loading ? <p className="emptyText">Cargando historial...</p> : null}
        {error ? <p className="errorText">{error}</p> : null}
        {!loading && !records.length ? <p className="emptyText">Sin inspecciones todavía.</p> : null}
        {records.map((record) => {
          const source = recordSource(record);
          const result = source.result as InspectionResult["result"] | undefined;
          const missing = Number(result?.piece_inspection?.missing_count || 0);
          return (
            <article className="historyRow" key={String(source.id || record.id)}>
              <span className={`resultBadge ${String(source.status || "idle")}`}>{statusLabel(String(source.status || "idle") as InspectionStatus)}</span>
              <div>
                <strong>{String(source.message || "Inspección registrada")}</strong>
                <small>{formatDate(String(source.created_at || record.created_at || ""))}</small>
              </div>
              <b>{missing} faltantes</b>
            </article>
          );
        })}
      </section>
    </>
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

function CaptureStillPanel({
  previewUrl,
  status,
  onOpen,
  onFile,
  onInspect
}: {
  previewUrl: string;
  status: InspectionStatus;
  onOpen: () => void;
  onFile: (file: File | null) => void;
  onInspect: () => void;
}) {
  return (
    <div className="captureStill">
      <div className="stillPreview">
        {previewUrl ? <img src={previewUrl} alt="Foto capturada" /> : <span>Sin foto capturada</span>}
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
  onFile,
  onCapture,
  onCancel,
  fullscreen = false
}: {
  family: string;
  zoneId: string;
  previewUrl: string;
  onFile: (file: File | null) => void;
  onCapture?: (file: File) => void | Promise<void>;
  onCancel?: () => void;
  fullscreen?: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const guidancePending = useRef(false);
  const stableReadyFrames = useRef(0);
  const autoCapturePending = useRef(false);
  const [active, setActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [guidance, setGuidance] = useState<CaptureGuidanceResult | null>(null);
  const [readyFrames, setReadyFrames] = useState(0);

  useEffect(() => {
    return () => stopCamera(streamRef);
  }, []);

  useEffect(() => {
    if (fullscreen && !active && !streamRef.current) {
      void startCamera();
    }
  }, [fullscreen]);

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => {
      void requestGuidance();
    }, 1400);
    return () => window.clearInterval(timer);
  }, [active, family, zoneId]);

  async function startCamera() {
    setCameraError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setActive(true);
    } catch (error) {
      setCameraError(messageFrom(error));
    }
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
      const response = (await postJson("/v1/capture-guidance", { family, zone_id: zoneId, image_uri: imageUri })) as CaptureGuidanceResult;
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

  const blockedByGuidance = active && (!guidance || !guidance.ok);
  const confidence = guidance?.alignment?.mold_segmentation?.confidence;
  const livePolygon = guidance?.alignment?.mold_segmentation?.polygon_normalized;
  const guidanceItems = guidance?.guidance?.length ? guidance.guidance : ["Centra el molde dentro del marco."];
  const readiness = Math.min(3, readyFrames);

  const stage = (
    <div className="dropzone cameraStage">
      {active ? <video ref={videoRef} playsInline muted /> : previewUrl ? <img src={previewUrl} alt="Vista previa" /> : <span>Toma o sube una foto de la zona</span>}
      {!fullscreen ? <input type="file" accept="image/*" capture="environment" onChange={(event) => onFile(event.target.files?.[0] ?? null)} /> : null}
      {active && livePolygon?.length ? (
        <svg className="liveMoldPolygon" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <polygon points={polygonPoints(livePolygon)} />
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
            <span>Usa el celular horizontal y coloca todo el molde dentro del marco.</span>
          </div>
          <button className="secondary" type="button" onClick={closeFullscreen}>Cancelar</button>
        </div>
        <div className="fullscreenStageWrap">{stage}</div>
        <div className={`fullscreenGuidance ${guidance?.ok ? "isOk" : "needsWork"}`}>
          <div>
            <strong>{cameraError || guidance?.message || "Detectando molde..."}</strong>
            <span>{typeof confidence === "number" ? `Confianza ${Math.round(confidence * 100)}%` : "Esperando detección"}</span>
          </div>
          <div className="readinessMeter" aria-label="estabilidad">
            {[0, 1, 2].map((item) => <i key={item} className={item < readiness ? "ready" : ""} />)}
          </div>
          <ul>{guidanceItems.map((item) => <li key={item}>{item}</li>)}</ul>
          <button className="primary" type="button" onClick={capturePhoto} disabled={Boolean(blockedByGuidance || cameraError)}>Capturar ahora</button>
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
  const resultImage = previewUrl || (result?.evidence?.overlay_image ? resolveUrl(result.evidence.overlay_image) : "");
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
  const dbRecords = await selectSupabase(table);
  if (dbRecords.length) return dbRecords;
  if (!endpoint) return [];
  const backendRecords = await getJson(endpoint).catch(() => []);
  return Array.isArray(backendRecords) ? backendRecords : [];
}

async function selectSupabase(table: string): Promise<ResourceRecord[]> {
  const { data, error } = await supabase.from(table).select("*").limit(100).order("created_at", { ascending: false });
  if (error) return [];
  return (data || []) as ResourceRecord[];
}

async function insertSupabase(table: string, payload: Record<string, unknown>) {
  try {
    await supabase.from(table).upsert(payload).select().maybeSingle();
  } catch {
    return;
  }
}

async function uploadFiles(files: File[], family: string, zoneId: string, purpose: string) {
  return Promise.all(files.map((file) => uploadSystemFile(file, family, zoneId, purpose)));
}

async function uploadSystemFile(file: File, family: string, zoneId: string, purpose: string) {
  const presign = await postJson("/v1/uploads/presign", {
    filename: file.name,
    content_type: file.type || "application/octet-stream",
    family,
    zone_id: zoneId,
    purpose
  });
  await fetch(resolveUrl(presign.upload_url), { method: presign.method, headers: presign.headers, body: file });
  return presign.object_uri as string;
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

function polygonPoints(points: Array<{ x: number; y: number }>) {
  return points.map((point) => `${point.x * 100},${point.y * 100}`).join(" ");
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function insertPointInClosestSegment(points: Point[], point: Point) {
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

function distanceToSegment(point: Point, start: Point, end: Point) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (dx === 0 && dy === 0) return Math.hypot(point.x - start.x, point.y - start.y);
  const t = clamp(((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy), 0, 1);
  return Math.hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy));
}

async function postJson(path: string, body: unknown) {
  const response = await fetch(resolveUrl(path), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
  return response.json();
}

async function getJson(path: string) {
  const response = await fetch(resolveUrl(path));
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
  return response.json();
}

function resolveUrl(path: string) {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
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

function messageFrom(error: unknown) {
  return error instanceof Error ? error.message : "Error inesperado.";
}

function statusLabel(status: InspectionStatus) {
  return { idle: "Esperando", uploading: "Subiendo", running: "Validando", correct: "Correcto", review: "Revisar", retake_photo: "Tomar otra foto", error: "Error" }[status];
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

  for (const recipe of recipes) {
    const family = recordString(recipe, "family") || fieldValue(recipe, "family") || String(recipe.id || "molde_demo");
    const zoneId = recordString(recipe, "zone_id") || DEFAULT_ZONES[0].id;
    ensure(family, zoneId, recordString(recipe, "name") || String(recipe.id));
  }

  for (const dataset of datasets) {
    const family = recordString(dataset, "family") || fieldValue(dataset, "family");
    const zoneId = recordString(dataset, "zone_id") || fieldValue(dataset, "zone_id") || DEFAULT_ZONES[0].id;
    if (!family) continue;
    const id = ensure(family, zoneId);
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
    const current = keys.get(id) || {};
    const jobStatus = String(source.status || "").toLowerCase();
    keys.set(id, {
      ...current,
      lastTraining: String(source.updated_at || source.created_at || job.updated_at || job.created_at || ""),
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
    const current = keys.get(id) || {};
    keys.set(id, { ...current, lastInspection: String(source.created_at || inspection.created_at || "") });
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

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
