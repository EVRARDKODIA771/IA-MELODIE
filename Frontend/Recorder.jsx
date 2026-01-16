import { useEffect, useRef, useState } from "react";
import "./Recorder.css";

export default function Recorder() {
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [time, setTime] = useState(0);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState(null);

  // Format mm:ss
  const formatTime = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  // Démarrer enregistrement
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
    setStatus("🎙️ Enregistrement en cours");

    timerRef.current = setInterval(() => {
      setTime(t => t + 1);
    }, 1000);
  };

  // Pause / reprise
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
      setStatus("🎙️ Enregistrement en cours");
      timerRef.current = setInterval(() => {
        setTime(t => t + 1);
      }, 1000);
    }
  };

  // Stop
  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
    clearInterval(timerRef.current);
    setIsRecording(false);
    setStatus("⏹️ Enregistrement terminé");
  };

  // Envoi au backend Render
  const sendAudio = async () => {
    if (!audioBlob) return;

    setStatus("📤 Envoi au serveur...");

    const formData = new FormData();
    formData.append("file", audioBlob, "recording.webm");

    await fetch("https://ia-melodie.onrender.com/melody/upload", {
      method: "POST",
      body: formData
    });

    setStatus("⏳ Analyse en cours...");

    // Récupération résultat
    const interval = setInterval(async () => {
      const res = await fetch("https://ia-melodie.onrender.com/melody/result");
      if (res.ok) {
        const json = await res.json();
        setResult(json.result);
        setStatus("✅ Analyse terminée");
        clearInterval(interval);
      }
    }, 2000);
  };

  return (
    <div className="recorder-container">
      <h1>🎵 Chanter ou enregistrer une musique </h1>

      <div className={`indicator ${isRecording ? "on" : ""}`} />

      <div className="time">{formatTime(time)}</div>
      <p className="status">{status}</p>

      <div className="buttons">
        {!isRecording && <button onClick={startRecording}>🎙️ Démarrer</button>}
        {isRecording && <button onClick={togglePause}>{isPaused ? "▶️ Reprendre" : "⏸️ Pause"}</button>}
        {isRecording && <button onClick={stopRecording}>⏹️ Stop</button>}
      </div>

      {audioBlob && (
        <>
          <audio controls src={URL.createObjectURL(audioBlob)} />
          <button className="send" onClick={sendAudio}>📤 Analyser la musique</button>
        </>
      )}

      {result && (
        <pre className="result">
{JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}
