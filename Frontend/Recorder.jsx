import { useRef, useState, useEffect } from "react";
import "./Recorder.css";

// ======================
// URL du backend depuis .env
// ======================
const backendUrl = import.meta.env.VITE_BACKEND_URL;

// Récupérer le returnUrl passé en query param (depuis Wix)
const urlParams = new URLSearchParams(window.location.search);
const returnUrl = urlParams.get("returnUrl");

// Fonction utilitaire pour générer un JobID unique
const generateJobId = () => "job-" + Math.random().toString(36).substr(2, 9);

export default function Recorder() {
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

  const formatTime = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  // ======================
  // PING BACKEND
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
  // START RECORDING (auto stop 7s)
  // ======================
  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);

    mediaRecorderRef.current = recorder;
    chunksRef.current = [];
    setAudioBlob(null);
    setResult(null);

    const newJobId = generateJobId();
    setJobId(newJobId);

    recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      setAudioBlob(blob);
      setStatus("🧠 Enregistrement terminé, envoi au serveur...");
      sendAudio(blob, newJobId);
    };

    recorder.start();
    setIsRecording(true);
    setIsPaused(false);
    setTime(0);
    setStatus("🎶 Enregistrement en cours...");

    timerRef.current = setInterval(() => setTime((t) => t + 1), 1000);

    setTimeout(() => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
        stopRecording();
      }
    }, 7000);
  };

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

  const stopRecording = () => {
    if (!mediaRecorderRef.current) return;
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    clearInterval(timerRef.current);
    setIsRecording(false);
  };

  // ======================
  // SEND AUDIO AUdD + redirect returnUrl
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

      // URL pour récupérer le JSON
      const resultUrl = `${backendUrl}/melody/result/${jobId}?backend=audd`;

      setStatus("✅ Musique envoyée ! Récupération en cours...");

      // Si returnUrl est défini, rediriger vers Wix avec jobId et resultUrl
      if (returnUrl) {
        const wixRedirect = `${decodeURIComponent(returnUrl)}?jobId=${jobId}&resultUrl=${encodeURIComponent(resultUrl)}`;
        window.location.href = wixRedirect;
      } else {
        setResult({ jobId, resultUrl });
      }
    } catch (err) {
      console.error(err);
      setStatus("❌ Erreur d'identification");
    }
  };

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
