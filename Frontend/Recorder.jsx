import { useRef, useState, useEffect } from "react";
import "./Recorder.css";

// ======================
// URL du backend depuis .env
// ======================
const backendUrl = import.meta.env.VITE_BACKEND_URL;

// IMPORTANT: ton Node API (fingerprint) est sur ia-melodie.onrender.com
// Si VITE_BACKEND_URL pointe déjà vers ia-melodie.onrender.com, tu peux laisser = backendUrl
// Sinon, garde ce fallback explicite.
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
  const [jobId, setJobId] = useState(null);

  // ======================
  // Récupérer jobId et returnUrl depuis query param
  // ======================
  const urlParams = new URLSearchParams(window.location.search);
  const jobIdFromWix = urlParams.get("jobId"); // jobId fourni par Wix
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
  // Polling fingerprint (route: /fingerprint/:jobId)
  // ======================
  const pollFingerprint = async (
    pollJobId,
    { interval = 2000, timeout = 120000 } = {}
  ) => {
    const start = Date.now();

    while (true) {
      if (Date.now() - start > timeout) throw new Error("Timeout fingerprint");

      const r = await fetch(`${apiUrl}/fingerprint/${pollJobId}`);
      const data = await r.json();

      // data = {status:"processing"/"done"/"error", jobId, resultUrl, fingerprint?}
      if (data.status === "done") return data;
      if (data.status === "error")
        throw new Error(data.message || "Fingerprint error");

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
  // SEND AUDIO : AUdD + Fingerprint en parallèle (sans casser l'un ou l'autre)
  // ======================
  const sendAudio = async (blob, baseJobId) => {
    if (!blob || !baseJobId) return;

    // 1) AUdD jobId = baseJobId
    const fdAudd = new FormData();
    fdAudd.append("file", blob, "recording.webm");
    fdAudd.append("jobId", baseJobId);

    // 2) Fingerprint jobId = baseJobId + "-fp" (évite collision)
    const fpJobId = `${baseJobId}-fp`;
    const fdFp = new FormData();
    fdFp.append("file", blob, "recording.webm");
    fdFp.append("jobId", fpJobId);

    setStatus("🧠 Envoi AUdD + Fingerprint en parallèle...");

    // Lance les deux requêtes en parallèle
    const [auddSettled, fpSettled] = await Promise.allSettled([
      fetch(`${apiUrl}/melody/upload?backend=audd`, {
        method: "POST",
        body: fdAudd,
      }),
      // IMPORTANT: dans ton server.js actuel, la route est /fingerprint/upload (pas /fingerprint/hum/upload)
      fetch(`${apiUrl}/fingerprint/upload`, {
        method: "POST",
        body: fdFp,
      }),
    ]);

    // --- AUdD ---
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

    // --- Fingerprint ---
    let fpOk = false;
    let fpFingerprint = null;
    let fpResultUrl = `${apiUrl}/fingerprint/result/${fpJobId}`;

    if (fpSettled.status === "fulfilled") {
      const fpRes = fpSettled.value;
      if (fpRes.ok) {
        fpOk = true;

        try {
          // 1) poll jusqu'à done
          const pollData = await pollFingerprint(fpJobId);

          // 2) fetch JSON final via resultUrl renvoyé
          const finalUrl = pollData.resultUrl
            ? `${apiUrl}${pollData.resultUrl}`
            : fpResultUrl;

          const finalRes = await fetch(finalUrl);
          const finalJson = await finalRes.json();

          // python renvoie: { fingerprint, fingerprint_short, meta }
          fpFingerprint = finalJson.fingerprint || pollData.fingerprint || null;

          // si jamais le resultUrl renvoyé est plus précis, on le garde
          fpResultUrl = finalUrl;
        } catch (e) {
          fpOk = false;
          console.error("Fingerprint polling/fetch error:", e);
        }
      } else {
        console.error("Fingerprint HTTP error:", fpRes.status);
      }
    } else {
      console.error("Fingerprint error:", fpSettled.reason);
    }

    // --- Status global ---
    if (auddOk && fpOk) setStatus("✅ AUdD + Fingerprint terminés");
    else if (auddOk) setStatus("⚠️ AUdD OK, Fingerprint échoué");
    else if (fpOk) setStatus("⚠️ Fingerprint OK, AUdD échoué");
    else setStatus("❌ AUdD + Fingerprint échoués");

    // --- UI result (debug) ---
    const out = {
      jobId: baseJobId,
      auddResultUrl,
      fpJobId,
      fpResultUrl,
      fpFingerprint,
    };
    setResult(out);

    // --- Redirection Wix ---
    // Même logique que AUdD : on renvoie AUSSI l'URL du JSON fingerprint
    if (returnUrl) {
      try {
        const wixUrl = new URL(decodeURIComponent(returnUrl));
        wixUrl.searchParams.set("jobId", baseJobId);

        // AUdD (comme avant)
        if (auddResultUrl) wixUrl.searchParams.set("resultUrl", auddResultUrl);

        // Fingerprint (nouveau, même philosophie)
        wixUrl.searchParams.set("fingerprintJobId", fpJobId);
        wixUrl.searchParams.set("fingerprintResultUrl", fpResultUrl);
        if (fpFingerprint) wixUrl.searchParams.set("fingerprint", fpFingerprint);

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

          <p>Fingerprint JobID : {result.fpJobId}</p>

          <p>
            Fingerprint JSON :{" "}
            <a
              href={result.fpResultUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              {result.fpResultUrl}
            </a>
          </p>

          {result.fpFingerprint && (
            <p>
              Empreinte musicale : <b>{result.fpFingerprint}</b>
            </p>
          )}
        </div>
      )}
    </div>
  );
}
