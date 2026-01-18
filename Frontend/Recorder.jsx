import { useRef, useState, useEffect } from "react";
import "./Recorder.css";

// ======================
// URL du backend depuis .env
// ======================
const backendUrl = import.meta.env.VITE_BACKEND_URL;

export default function Recorder() {

  // ======================
  // PARAMÈTRES REÇUS DE WIX
  // ======================
  const urlParams = new URLSearchParams(window.location.search);
  const jobId = urlParams.get("jobId");
  const userId = urlParams.get("user");
  const returnUrl = urlParams.get("returnUrl");

  // Sécurité minimale
  if (!jobId || !userId || !returnUrl) {
    return (
      <div style={{ padding: 20 }}>
        ❌ Paramètres manquants (jobId / user / returnUrl)
      </div>
    );
  }

  // ======================
  // ANTI-SOMMEIL / PING BACKEND
  // ======================
  useEffect(() => {
    fetch(`${backendUrl}/ping`).catch(() => {});
  }, []);

  // ======================
  // STATES
  // ======================
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [time, setTime] = useState(0);
  const [status, setStatus] = useState("Touchez le micro pour chanter");

  const formatTime = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  // ======================
  // START RECORDING
  // ======================
  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);

    mediaRecorderRef.current = recorder;
    chunksRef.current = [];

    recorder.ondataavailable = (e) => chunksRef.current.push(e.data);

    recorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      setStatus("🧠 Envoi au serveur...");
      await sendAudio(blob);
    };

    recorder.start();
    setIsRecording(true);
    setIsPaused(false);
    setTime(0);
    setStatus("🎶 Enregistrement en cours...");

    timerRef.current = setInterval(() => setTime(t => t + 1), 1000);

    setTimeout(stopRecording, 7000);
  };

  // ======================
  // STOP RECORDING
  // ======================
  const stopRecording = () => {
    if (!mediaRecorderRef.current) return;
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
    clearInterval(timerRef.current);
    setIsRecording(false);
  };

  // ======================
  // PAUSE / RESUME
  // ======================
  const togglePause = () => {
    const rec = mediaRecorderRef.current;
    if (!rec) return;

    if (rec.state === "recording") {
      rec.pause();
      setIsPaused(true);
      setStatus("⏸️ En pause");
      clearInterval(timerRef.current);
    } else {
      rec.resume();
      setIsPaused(false);
      setStatus("🎶 Enregistrement en cours...");
      timerRef.current = setInterval(() => setTime(t => t + 1), 1000);
    }
  };

  // ======================
  // SEND AUDIO → BACKEND
  // ======================
  const sendAudio = async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "recording.webm");
    formData.append("jobId", jobId);
    formData.append("user", userId);

    try {
      const res = await fetch(`${backendUrl}/melody/upload?backend=audd`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Erreur serveur");

      setStatus("✅ Identification terminée, retour vers Wix...");

      // ======================
      // RETOUR EXACT VERS WIX
      // ======================
      const wixUrl = new URL(decodeURIComponent(returnUrl));
      wixUrl.searchParams.set("jobId", jobId);
      wixUrl.searchParams.set("user", userId);

      window.location.href = wixUrl.toString();

    } catch (err) {
      console.error(err);
      setStatus("❌ Erreur d'identification");
    }
  };

  // ======================
  // UI
  // ======================
  return (
    <div className="recorder-container">
      <div className="title">PARTITION MANAGER</div>
      <div className="subtitle">Chantez ou fredonnez une musique</div>

      <div
        className="pulse-wrapper"
        onClick={!isRecording ? startRecording : stopRecording}
      >
        {isRecording && !isPaused && <div className="pulse" />}
        {isRecording && !isPaused && <div className="pulse delay" />}
        <div className="center-circle">🎤</div>
      </div>

      <div className="time">{formatTime(time)}</div>
      <div className="status">{status}</div>

      {isRecording && (
        <div className="buttons">
          <button onClick={togglePause}>
            {isPaused ? "▶️ Reprendre" : "⏸️ Pause"}
          </button>
        </div>
      )}
    </div>
  );
}
