import { useRef, useState, useEffect } from "react";
import "./Recorder.css";

// ======================
// URL du backend depuis .env
// ======================
const backendUrl = import.meta.env.VITE_BACKEND_URL;

export default function Recorder() {
  // ======================
  // ANTI-SOMMEIL / PING BACKEND
  // ======================
  useEffect(() => {
    const pingBackend = async () => {
      try {
        await fetch(`${backendUrl}/ping`);
      } catch (err) {
        console.error("Ping backend failed", err);
      }
    };

    pingBackend();
    const interval = setInterval(pingBackend, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // ======================
  // STATES
  // ======================
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [time, setTime] = useState(0);
  const [status, setStatus] = useState("Touchez le micro pour chanter");
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);

  // ======================
  // Récupérer jobId et returnUrl depuis query param
  // ======================
  const urlParams = new URLSearchParams(window.location.search);
  const jobIdFromWix = urlParams.get("jobId"); // Utiliser le jobId fourni par Wix
  const returnUrl = urlParams.get("returnUrl");

  useEffect(() => {
    if (jobIdFromWix) setJobId(jobIdFromWix);
  }, [jobIdFromWix]);

  const formatTime = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  // ======================
  // START RECORDING (auto stop 7s)
  // ======================
  const startRecording = async () => {
    if (!jobIdFromWix) {
      console.error("❌ Pas de jobId reçu depuis Wix");
      setStatus("❌ Erreur : jobId manquant");
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);

    mediaRecorderRef.current = recorder;
    chunksRef.current = [];
    setAudioBlob(null);
    setResult(null);

    recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      setAudioBlob(blob);
      setStatus("🧠 Enregistrement terminé, envoi au serveur...");
      sendAudio(blob, jobIdFromWix);
    };

    recorder.start();
    setIsRecording(true);
    setIsPaused(false);
    setTime(0);
    setStatus("🎶 Enregistrement en cours...");

    // Timer
    timerRef.current = setInterval(() => setTime((t) => t + 1), 1000);

    // Auto-stop après 7 secondes
    setTimeout(() => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
        stopRecording();
      }
    }, 7000);
  };

  // ======================
  // PAUSE / RESUME
  // ======================
  const togglePause = () => {
    if (!mediaRecorderRef.current) return;

    if (mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.pause();
      setIsPaused(true);
      setStatus("⏸️ En pause");
      clearInterval(timerRef.current);
    } else {
      mediaRecorderRef.current.resume();
      setIsPaused(false);
      setStatus("🎶 Enregistrement en cours...");
      timerRef.current = setInterval(() => setTime((t) => t + 1), 1000);
    }
  };

  // ======================
  // STOP RECORDING
  // ======================
  const stopRecording = () => {
    if (!mediaRecorderRef.current) return;
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    clearInterval(timerRef.current);
    setIsRecording(false);
  };

  // ======================
  // SEND AUDIO AUdD
  // ======================
  const sendAudio = async (blob, jobId) => {
    if (!blob || !jobId) return;

    const formData = new FormData();
    formData.append("file", blob, "recording.webm");
    formData.append("jobId", jobId);

    try {
      const res = await fetch(`${backendUrl}/melody/upload?backend=audd`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Erreur serveur");

      const resultUrl = `${backendUrl}/melody/result/${jobId}?backend=audd`;
      setStatus(`✅ Musique envoyée !`);

      setResult({ jobId, resultUrl });

      // ======================
      // REDIRECTION WIX AVEC JOBID ET RESULTURL
      // ======================
      if (returnUrl) {
        try {
          const wixUrl = new URL(decodeURIComponent(returnUrl));
          wixUrl.searchParams.set("jobId", jobId); // exact jobId de Wix
          wixUrl.searchParams.set("resultUrl", resultUrl);
          window.location.href = wixUrl.toString();
        } catch (err) {
          console.error("Erreur parsing returnUrl Wix :", err);
        }
      }
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

      {/* SHAZAM CIRCLE */}
      <div
        className="pulse-wrapper"
        onClick={!isRecording ? startRecording : stopRecording}
      >
        {isRecording && !isPaused && <div className="pulse" />}
        {isRecording && !isPaused && <div className="pulse delay" />}
        <div className="center-circle">🎤</div>
      </div>

      {/* TIMER */}
      <div className="time">{formatTime(time)}</div>

      {/* STATUS */}
      <div className="status">{status}</div>

      {/* CONTROLS */}
      {isRecording && (
        <div className="buttons">
          <button onClick={togglePause}>
            {isPaused ? "▶️ Reprendre" : "⏸️ Pause"}
          </button>
        </div>
      )}

      {/* RESULT */}
      {result && (
        <div className="result">
          <p>JobID : {result.jobId}</p>
          <p>
            Récupérer le résultat via :{" "}
            <a href={result.resultUrl} target="_blank" rel="noopener noreferrer">
              {result.resultUrl}
            </a>
          </p>
        </div>
      )}
    </div>
  );
}
