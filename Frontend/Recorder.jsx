import { useRef, useState, useEffect } from "react";
import "./Recorder.css";

// ======================
// URL du backend depuis .env
// ======================
const backendUrl = import.meta.env.VITE_BACKEND_URL;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(
      2,
      "0"
    )}`;

  // ======================
  // Polling fingerprint HUM
  // ======================
  const pollFingerprint = async (pollJobId, { interval = 2000, timeout = 120000 } = {}) => {
    const start = Date.now();
    while (true) {
      if (Date.now() - start > timeout) throw new Error("Timeout HUM");

      const r = await fetch(`${backendUrl}/fingerprint/${pollJobId}`);
      const data = await r.json();

      // data = {status:"processing"/"done"/"error", jobId, resultUrl}
      if (data.status === "done") return data;
      if (data.status === "error") throw new Error(data.message || "HUM error");

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

    // Timer
    timerRef.current = setInterval(() => setTime((t) => t + 1), 1000);

    // Auto-stop après 7 secondes
    setTimeout(() => {
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state === "recording"
      ) {
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
  // SEND AUDIO : AUdD + HUM en parallèle (sans casser l'un ou l'autre)
  // ======================
  const sendAudio = async (blob, baseJobId) => {
    if (!blob || !baseJobId) return;

    // 1) AUdD jobId = baseJobId
    const fdAudd = new FormData();
    fdAudd.append("file", blob, "recording.webm");
    fdAudd.append("jobId", baseJobId);

    // 2) HUM jobId = baseJobId + "-hum" (évite collision)
    const humJobId = `${baseJobId}-hum`;
    const fdHum = new FormData();
    fdHum.append("file", blob, "recording.webm");
    fdHum.append("jobId", humJobId);

    setStatus("🧠 Envoi AUdD + HUM en parallèle...");

    // Lance les deux requêtes en parallèle
    const [auddSettled, humSettled] = await Promise.allSettled([
      fetch(`${backendUrl}/melody/upload?backend=audd`, {
        method: "POST",
        body: fdAudd,
      }),
      fetch(`${backendUrl}/fingerprint/hum/upload`, {
        method: "POST",
        body: fdHum,
      }),
    ]);

    // --- AUdD ---
    let auddOk = false;
    let auddResultUrl = null;

    if (auddSettled.status === "fulfilled") {
      const auddRes = auddSettled.value;
      if (auddRes.ok) {
        auddOk = true;
        auddResultUrl = `${backendUrl}/melody/result/${baseJobId}?backend=audd`;
      } else {
        console.error("AUdD HTTP error:", auddRes.status);
      }
    } else {
      console.error("AUdD error:", auddSettled.reason);
    }

    // --- HUM ---
    let humOk = false;
    let humFingerprint = null;
    let humResultUrl = `${backendUrl}/fingerprint/result/${humJobId}`;

    if (humSettled.status === "fulfilled") {
      const humRes = humSettled.value;
      if (humRes.ok) {
        humOk = true;

        try {
          // On poll jusqu'à done
          const pollData = await pollFingerprint(humJobId);

          // pollData.resultUrl est un chemin "/fingerprint/result/..."
          const finalRes = await fetch(`${backendUrl}${pollData.resultUrl}`);
          const finalJson = await finalRes.json();

          // Ton python actuel renvoie: { fingerprint, fingerprint_short, meta, ... }
          humFingerprint = finalJson.fingerprint || null;
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

    // --- Status global ---
    if (auddOk && humOk) setStatus("✅ AUdD + HUM terminés");
    else if (auddOk) setStatus("⚠️ AUdD OK, HUM échoué");
    else if (humOk) setStatus("⚠️ HUM OK, AUdD échoué");
    else setStatus("❌ AUdD + HUM échoués");

    // --- UI result (debug) ---
    const out = {
      jobId: baseJobId,
      auddResultUrl,
      humJobId,
      humResultUrl,
      humFingerprint,
    };
    setResult(out);

    // --- Redirection Wix ---
    // On renvoie les 2 urls pour ne rien casser + garder une trace HUM
    if (returnUrl) {
      try {
        const wixUrl = new URL(decodeURIComponent(returnUrl));
        wixUrl.searchParams.set("jobId", baseJobId);

        if (auddResultUrl) wixUrl.searchParams.set("resultUrl", auddResultUrl);

        // AJOUTS HUM (non bloquants)
        wixUrl.searchParams.set("humJobId", humJobId);
        wixUrl.searchParams.set("humResultUrl", humResultUrl);
        if (humFingerprint) wixUrl.searchParams.set("humFingerprint", humFingerprint);

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

          {result.auddResultUrl && (
            <p>
              AUdD résultat :{" "}
              <a
                href={result.auddResultUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                {result.auddResultUrl}
              </a>
            </p>
          )}

          <p>HUM JobID : {result.humJobId}</p>
          <p>
            HUM JSON :{" "}
            <a href={result.humResultUrl} target="_blank" rel="noopener noreferrer">
              {result.humResultUrl}
            </a>
          </p>

          {result.humFingerprint && (
            <p>
              HUM fingerprint : <b>{result.humFingerprint}</b>
            </p>
          )}
        </div>
      )}
    </div>
  );
}
