import { useRef, useState, useEffect } from "react";
import "./Recorder.css";

// URL backend depuis .env (doit être https://ia-melodie.onrender.com)
const backendUrl = import.meta.env.VITE_BACKEND_URL;

// fallback si jamais .env est vide (évite d'envoyer vers ia-melodie-1 par erreur)
const apiUrl = backendUrl || "https://ia-melodie.onrender.com";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export default function Recorder() {
  // ======================
  // ANTI-SOMMEIL / PING BACKEND
  // ======================
  useEffect(() => {
    const pingBackend = async () => {
      try {
        await fetch(`${apiUrl}/ping`);
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

  // ======================
  // Query params Wix
  // ======================
  const urlParams = new URLSearchParams(window.location.search);
  const jobIdFromWix = urlParams.get("jobId");
  const returnUrl = urlParams.get("returnUrl");

  const formatTime = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(
      2,
      "0"
    )}`;

  // ======================
  // Polling générique (utilise /fingerprint/:jobId)
  // ======================
  const pollJob = async (pollJobId, { interval = 1500, timeout = 120000 } = {}) => {
    const start = Date.now();

    while (true) {
      if (Date.now() - start > timeout) throw new Error("Timeout polling");

      const r = await fetch(`${apiUrl}/fingerprint/${pollJobId}`);
      const data = await r.json();

      if (data.status === "done") return data;
      if (data.status === "error") throw new Error(data.message || "Job error");

      await sleep(interval);
    }
  };

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

    timerRef.current = setInterval(() => setTime((t) => t + 1), 1000);

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
  // SEND AUDIO : AUdD + HUM en parallèle
  // ======================
  const sendAudio = async (blob, baseJobId) => {
    if (!blob || !baseJobId) return;

    // AUdD
    const fdAudd = new FormData();
    fdAudd.append("file", blob, "recording.webm");
    fdAudd.append("jobId", baseJobId);

    // HUM (IMPORTANT)
    const humJobId = `${baseJobId}-hum`;
    const fdHum = new FormData();
    fdHum.append("file", blob, "recording.webm");
    fdHum.append("jobId", humJobId);

    setStatus("🧠 Envoi AUdD + HUM en parallèle...");

    const [auddSettled, humSettled] = await Promise.allSettled([
      fetch(`${apiUrl}/melody/upload?backend=audd`, {
        method: "POST",
        body: fdAudd,
      }),
      fetch(`${apiUrl}/fingerprint/hum/upload`, {
        method: "POST",
        body: fdHum,
      }),
    ]);

    // ===== AUdD 결과 =====
    let auddOk = false;
    let auddResultUrl = null;

    if (auddSettled.status === "fulfilled") {
      const auddRes = auddSettled.value;
      if (auddRes.ok) {
        auddOk = true;
        auddResultUrl = `${apiUrl}/melody/result/${baseJobId}?backend=audd`;
      } else {
        console.error("AUdD HTTP error:", auddRes.status);
      }
    } else {
      console.error("AUdD error:", auddSettled.reason);
    }

    // ===== HUM 결과 =====
    let humOk = false;
    let humResultUrl = `${apiUrl}/fingerprint/result/${humJobId}`;

    // Champs HUM utiles
    let humSignatureOk = false;
    let humMelodyHash = null;
    let humVoicedRatio = null;
    let humSignatureLen = null;

    if (humSettled.status === "fulfilled") {
      const humRes = humSettled.value;
      if (humRes.ok) {
        humOk = true;

        try {
          // 1) attendre done
          const pollData = await pollJob(humJobId);

          // 2) fetch json final
          const finalUrl = pollData.resultUrl
            ? `${apiUrl}${pollData.resultUrl}`
            : humResultUrl;

          const finalRes = await fetch(finalUrl);
          const finalJson = await finalRes.json();

          // server.js HUM met: { status:"done", jobId, hum:true, melody:{...}, meta:{...} }
          const melody = finalJson?.melody || null;

          humResultUrl = finalUrl;

          humSignatureOk = Boolean(melody?.melody_ok);
          humMelodyHash = melody?.melody_hash || null;
          humVoicedRatio = melody?.voiced_ratio ?? null;

          // signature = liste d'intervalles (120) si OK
          if (Array.isArray(melody?.signature)) {
            humSignatureLen = melody.signature.length;
          }
        } catch (e) {
          humOk = false;
          console.error("HUM polling/fetch error:", e);
        }
      } else {
        console.error("HUM HTTP error:", humRes.status);
      }
    } else {
      console.error("HUM error:", humSettled.reason);
    }

    // ===== Status UI =====
    if (auddOk && humOk) setStatus("✅ AUdD + HUM terminés");
    else if (auddOk) setStatus("⚠️ AUdD OK, HUM échoué");
    else if (humOk) setStatus("⚠️ HUM OK, AUdD échoué");
    else setStatus("❌ AUdD + HUM échoués");

    // ===== Debug UI =====
    const out = {
      jobId: baseJobId,
      auddResultUrl,
      humJobId,
      humResultUrl,
      humSignatureOk,
      humMelodyHash,
      humVoicedRatio,
      humSignatureLen,
    };
    setResult(out);

    // ===== Retour Wix =====
    // On renvoie comme AUdD + champs HUM
    if (returnUrl) {
      try {
        const wixUrl = new URL(decodeURIComponent(returnUrl));
        wixUrl.searchParams.set("jobId", baseJobId);

        // AUdD (inchangé)
        if (auddResultUrl) wixUrl.searchParams.set("resultUrl", auddResultUrl);

        // HUM (nouveau)
        wixUrl.searchParams.set("humJobId", humJobId);
        wixUrl.searchParams.set("humResultUrl", humResultUrl);

        // petits champs utiles pour debug côté Wix
        wixUrl.searchParams.set("humSignatureOk", String(humSignatureOk));
        if (humMelodyHash) wixUrl.searchParams.set("humMelodyHash", humMelodyHash);
        if (humVoicedRatio != null) wixUrl.searchParams.set("humVoicedRatio", String(humVoicedRatio));
        if (humSignatureLen != null) wixUrl.searchParams.set("humSignatureLen", String(humSignatureLen));

        window.location.href = wixUrl.toString();
      } catch (err) {
        console.error("Erreur parsing returnUrl Wix :", err);
      }
    }
  };

  // ======================
  // UI
  // ======================
  return (
    <div className="recorder-container">
      <div className="title">PARTITION MANAGER</div>
      <div className="subtitle">Chantez ou fredonnez une musique</div>

      <div className="pulse-wrapper" onClick={!isRecording ? startRecording : stopRecording}>
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

          {result.auddResultUrl && (
            <p>
              AUdD résultat :{" "}
              <a href={result.auddResultUrl} target="_blank" rel="noopener noreferrer">
                {result.auddResultUrl}
              </a>
            </p>
          )}

          <hr />

          <p>HUM JobID : {result.humJobId}</p>
          <p>
            HUM JSON :{" "}
            <a href={result.humResultUrl} target="_blank" rel="noopener noreferrer">
              {result.humResultUrl}
            </a>
          </p>

          <p>
            HUM melody_ok : <b>{String(result.humSignatureOk)}</b>
          </p>

          {result.humMelodyHash && (
            <p>
              HUM melody_hash : <b>{result.humMelodyHash}</b>
            </p>
          )}

          {result.humVoicedRatio != null && (
            <p>
              HUM voiced_ratio : <b>{String(result.humVoicedRatio)}</b>
            </p>
          )}

          {result.humSignatureLen != null && (
            <p>
              HUM signature_len : <b>{String(result.humSignatureLen)}</b>
            </p>
          )}
        </div>
      )}
    </div>
  );
}
