import { useRef, useState } from "react";
import "./Recorder.css";
import { useEffect } from "react";

useEffect(() => {
  const pingBackend = async () => {
    try {
      await fetch("https://ia-melodie.onrender.com/ping");
    } catch (err) {
      console.error("Ping backend failed", err);
    }
  };

  // Ping immédiat au chargement
  pingBackend();

  // Ping toutes les 5 minutes
  const interval = setInterval(pingBackend, 5 * 60 * 1000);

  // Cleanup à la destruction du composant
  return () => clearInterval(interval);
}, []);

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

  // Format mm:ss
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

    recorder.ondataavailable = e => chunksRef.current.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      setAudioBlob(blob);
    };

    recorder.start();
    setIsRecording(true);
    setIsPaused(false);
    setTime(0);
    setStatus("🎶 Enregistrement en cours...");

    timerRef.current = setInterval(() => setTime(t => t + 1), 1000);
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
      timerRef.current = setInterval(() => setTime(t => t + 1), 1000);
    }
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
    setStatus("🧠 Prêt pour l’analyse");
  };

  // ======================
  // SEND TO BACKEND AUdD
  // ======================
  const sendAudio = async () => {
    if (!audioBlob) return;

    setStatus("📤 Envoi au serveur...");

    const formData = new FormData();
    formData.append("file", audioBlob, "recording.webm");

    try {
      const res = await fetch("https://ia-melodie.onrender.com/melody/upload?backend=audd", {
        method: "POST",
        body: formData
      });

      if (!res.ok) throw new Error("Erreur serveur");

      const json = await res.json();
      setResult(json.result);
      setStatus("✅ Musique identifiée !");
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

      {/* SEND */}
      {audioBlob && !isRecording && (
        <button className="send" onClick={sendAudio}>
          🔍 Identifier la musique
        </button>
      )}

      {/* RESULT */}
      {result && (
        <pre className="result">{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}
