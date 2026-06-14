// Guided capture: live camera with a translucent ghost of the reference, a
// client-side alignment meter (edge correlation vs the reference within the
// zone), and hands-free auto-capture once the framing matches and holds steady.

import React, { useEffect, useRef, useState } from "react";
import { align, imageDataToGray, loadGray, type GrayImage, type Poly } from "./vision";

type Props = {
  referenceUrl: string;
  zone: Poly | null;
  onCapture: (file: File) => void;
  onCancel: () => void;
};

const GUIDE_W = 260;
const READY_THRESHOLD_FRAMES = 4;

export default function GuidedCapture({ referenceUrl, zone, onCapture, onCancel }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const guideCanvas = useRef<HTMLCanvasElement | null>(null);
  const fullCanvas = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const refGray = useRef<GrayImage | null>(null);
  const readyCount = useRef(0);
  const busy = useRef(false);
  const captured = useRef(false);

  const [error, setError] = useState("");
  const [opacity, setOpacity] = useState(0.45);
  const [strictness, setStrictness] = useState(0.6); // 0..1 -> threshold 0.45..0.78
  const [score, setScore] = useState(0);
  const [ready, setReady] = useState(0);
  const [guideH, setGuideH] = useState(Math.round(GUIDE_W / 1.5));

  const threshold = 0.45 + strictness * 0.33;

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
          audio: false,
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "No se pudo abrir la cámara");
      }
    })();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  // Precompute reference guidance gray/edge at a small fixed size.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const img = new Image();
      img.onload = async () => {
        if (cancelled) return;
        const h = Math.round(GUIDE_W / (img.naturalWidth / img.naturalHeight));
        setGuideH(h);
        refGray.current = await loadGray(referenceUrl, GUIDE_W);
      };
      img.src = referenceUrl;
    })();
    return () => { cancelled = true; };
  }, [referenceUrl]);

  // Guidance loop.
  useEffect(() => {
    const timer = window.setInterval(() => { void tick(); }, 320);
    return () => window.clearInterval(timer);
  }, [zone, threshold]);

  async function tick() {
    if (busy.current || captured.current) return;
    const video = videoRef.current;
    const ref = refGray.current;
    const canvas = guideCanvas.current;
    if (!video || !ref || !canvas || video.readyState < 2) return;
    busy.current = true;
    try {
      canvas.width = ref.width;
      canvas.height = ref.height;
      const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(video, 0, 0, ref.width, ref.height);
      const id = ctx.getImageData(0, 0, ref.width, ref.height);
      const insp = imageDataToGray(id.data, ref.width, ref.height);
      const result = align(ref, insp, zone);
      setScore(result.score);
      if (result.score >= threshold) readyCount.current += 1;
      else readyCount.current = 0;
      setReady(readyCount.current);
      if (readyCount.current >= READY_THRESHOLD_FRAMES) {
        await doCapture();
      }
    } finally {
      busy.current = false;
    }
  }

  async function doCapture() {
    if (captured.current) return;
    const video = videoRef.current;
    const canvas = fullCanvas.current;
    if (!video || !canvas) return;
    captured.current = true;
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, "image/jpeg", 0.92));
    if (blob) onCapture(new File([blob], `captura_${Date.now()}.jpg`, { type: "image/jpeg" }));
  }

  const pct = Math.round(score * 100);
  const aligned = score >= threshold;
  const progress = Math.min(1, ready / READY_THRESHOLD_FRAMES);

  return (
    <div className="labCapture" role="dialog" aria-modal="true">
      <div className="labCaptureTop">
        <strong>Captura guiada</strong>
        <button className="secondary" onClick={onCancel} type="button">Cancelar</button>
      </div>

      <div className="labCaptureStage">
        <video ref={videoRef} playsInline muted />
        <img className="labGhost" src={referenceUrl} alt="referencia" style={{ opacity }} />
        {zone && zone.length >= 3 ? (
          <svg className="labGhostZone" viewBox="0 0 100 100" preserveAspectRatio="none">
            <polygon points={zone.map((p) => `${p[0] * 100},${p[1] * 100}`).join(" ")} />
          </svg>
        ) : null}
        <div className={`labAlignRing ${aligned ? "ok" : ""}`} style={{ ["--p" as string]: progress }}>
          <span>{pct}%</span>
        </div>
        {error ? <div className="labCaptureError">{error}</div> : null}
      </div>

      <div className={`labCaptureControls ${aligned ? "ok" : ""}`}>
        <div className={`labAlignMsg ${aligned ? "ok" : ""}`}>
          {aligned ? "Encuadre correcto — mantén firme…" : "Mueve la tablet para que la pieza coincida con la guía"}
        </div>
        <div className="labCaptureSliders">
          <label>Transparencia guía
            <input type="range" min={15} max={80} value={Math.round(opacity * 100)} onChange={(e) => setOpacity(Number(e.target.value) / 100)} />
          </label>
          <label>Exigencia de encuadre
            <input type="range" min={0} max={100} value={Math.round(strictness * 100)} onChange={(e) => setStrictness(Number(e.target.value) / 100)} />
          </label>
        </div>
        <button className="primary" onClick={doCapture} type="button">Capturar ahora</button>
      </div>

      <canvas ref={guideCanvas} hidden />
      <canvas ref={fullCanvas} hidden />
    </div>
  );
}
