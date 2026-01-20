import { useRef, useState, useEffect } from "react";
import "./Recorder.css";

const backendUrl = import.meta.env.VITE_BACKEND_URL;
const apiUrl = backendUrl || "https://ia-melodie.onrender.com";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export default function Recorder() {
  useEffect(() => {
    const pingBackend = async () => {
      try {
        await fetch(`${apiUrl}/ping`, { cache: "no-store" });
      } catch (err) {
        console.error("Ping backend failed", err);
      }
    };

    pingBackend();
    const interval = setInterval(pingBackend, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [time, setTime] = useState(0);

  const [status, setStatus] = useState("Touchez le micro pour chanter");
  const [result, setResult] = useState(null);

  const urlParams = new URLSearchParams(window.location.search);
  const jobIdFromWix = urlParams.get("jobId");
  const returnUrl = urlParams.get("returnUrl");

  const formatTime = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  const pollJob = async (pollJobId, { interval = 1500, timeout = 120000 } = {}) => {
    const start = Date.now();

    while (true) {
      if (Date.now() - start > timeout) throw new Error("Timeout polling");

      const r = await fetch(`${apiUrl}/fingerprint/${pollJobId}`, { cache: "no-store" });
      const data = await r.json();

      if (data.status === "done") return data;
      if (data.status === "error") throw new Error(data.message || "Job error");

      await sleep(interval);
    }
  };

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
    setResult(null);

    recorder.ondataavailable = (e) => chunksRef.current.push(e.data);

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
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

  async function readJsonSafe(res) {
    try {
      return await res.json();
    } catch {
      return null;
    }
  }

  const sendAudio = async (blob, baseJobId) => {
    if (!blob || !baseJobId) return;

    const fdAudd = new FormData();
    fdAudd.append("file", blob, "recording.webm");
    fdAudd.append("jobId", baseJobId);

    const humJobId = `${baseJobId}-hum`;
    const fdHum = new FormData();
    fdHum.append("file", blob, "recording.webm");
    fdHum.append("jobId", humJobId);

    const auddUploadUrl = `${apiUrl}/melody/upload?backend=audd`;
    const humUploadUrl = `${apiUrl}/fingerprint/hum/upload`;

    console.log("➡️ AUdD upload =>", auddUploadUrl, baseJobId);
    console.log("➡️ HUM upload  =>", humUploadUrl, humJobId);

    setStatus("🧠 Envoi AUdD + HUM en parallèle...");

    const [auddSettled, humSettled] = await Promise.allSettled([
      fetch(auddUploadUrl, { method: "POST", body: fdAudd }),
      fetch(humUploadUrl, { method: "POST", body: fdHum }),
    ]);

    let auddOk = false;
    let auddResultUrl = null;
    let auddUploadInfo = null;

    if (auddSettled.status === "fulfilled") {
      const res = auddSettled.value;
      const json = await readJsonSafe(res);
      auddUploadInfo = { ok: res.ok, status: res.status, json };

      if (res.ok) {
        auddOk = true;
        auddResultUrl = `${apiUrl}/melody/result/${baseJobId}`;
      } else {
        console.error("AUdD HTTP error:", res.status, json);
      }
    } else {
      console.error("AUdD error:", auddSettled.reason);
      auddUploadInfo = { error: String(auddSettled.reason) };
    }

    let humOk = false;
    let humResultUrl = `${apiUrl}/fingerprint/result/${humJobId}`;
    let humLogsUrl = `${apiUrl}/fingerprint/logs/${humJobId}`;
    let humUploadInfo = null;

    let humSignatureOk = false;
    let humMelodyHash = null;
    let humVoicedRatio = null;
    let humSignatureLen = null;

    if (humSettled.status === "fulfilled") {
      const res = humSettled.value;
      const json = await readJsonSafe(res);
      humUploadInfo = { ok: res.ok, status: res.status, json };

      if (res.ok) {
        humOk = true;

        // si le backend renvoie logsUrl/resultUrl, on les utilise
        if (json?.resultUrl) humResultUrl = `${apiUrl}${json.resultUrl}`;
        if (json?.logsUrl) humLogsUrl = `${apiUrl}${json.logsUrl}`;

        try {
          const pollData = await pollJob(humJobId);
          const finalUrl = pollData?.resultUrl ? `${apiUrl}${pollData.resultUrl}` : humResultUrl;

          const finalRes = await fetch(finalUrl, { cache: "no-store" });
          const finalJson = await finalRes.json();

          humResultUrl = finalUrl;

          const melody = finalJson?.melody || null;
          humSignatureOk = Boolean(melody?.melody_ok);
          humMelodyHash = melody?.melody_hash || null;
          humVoicedRatio = melody?.voiced_ratio ?? null;
          if (Array.isArray(melody?.signature)) humSignatureLen = melody.signature.length;
        } catch (e) {
          humOk = false;
          console.error("HUM polling/fetch error:", e);
        }
      } else {
        console.error("HUM HTTP error:", res.status, json);
      }
    } else {
      console.error("HUM error:", humSettled.reason);
      humUploadInfo = { error: String(humSettled.reason) };
    }

    if (auddOk && humOk) setStatus("✅ AUdD + HUM terminés");
    else if (auddOk) setStatus("⚠️ AUdD OK, HUM échoué");
    else if (humOk) setStatus("⚠️ HUM OK, AUdD échoué");
    else setStatus("❌ AUdD + HUM échoués");

    const out = {
      jobId: baseJobId,
      auddUploadUrl,
      auddResultUrl,
      auddUploadInfo,

      humJobId,
      humUploadUrl,
      humResultUrl,
      humLogsUrl,
      humUploadInfo,

      humSignatureOk,
      humMelodyHash,
      humVoicedRatio,
      humSignatureLen,
    };
    setResult(out);

    // Retour Wix
    if (returnUrl) {
      try {
        const wixUrl = new URL(decodeURIComponent(returnUrl));
        wixUrl.searchParams.set("jobId", baseJobId);

        if (auddResultUrl) wixUrl.searchParams.set("resultUrl", auddResultUrl);

        wixUrl.searchParams.set("humJobId", humJobId);
        wixUrl.searchParams.set("humResultUrl", humResultUrl);
        wixUrl.searchParams.set("humLogsUrl", humLogsUrl);

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
          <button onClick={togglePause}>{isPaused ? "▶️ Reprendre" : "⏸️ Pause"}</button>
        </div>
      )}

      {result && (
        <div className="result">
          <p><b>JobID :</b> {result.jobId}</p>

          <p><b>AUdD upload :</b> {result.auddUploadUrl}</p>
          {result.auddResultUrl && (
            <p>
              <b>AUdD résultat :</b>{" "}
              <a href={result.auddResultUrl} target="_blank" rel="noopener noreferrer">
                {result.auddResultUrl}
              </a>
            </p>
          )}
          <pre style={{ whiteSpace: "pre-wrap" }}>
            {JSON.stringify(result.auddUploadInfo, null, 2)}
          </pre>

          <hr />

          <p><b>HUM JobID :</b> {result.humJobId}</p>
          <p><b>HUM upload :</b> {result.humUploadUrl}</p>

          <p>
            <b>HUM JSON :</b>{" "}
            <a href={result.humResultUrl} target="_blank" rel="noopener noreferrer">
              {result.humResultUrl}
            </a>
          </p>

          <p>
            <b>HUM logs :</b>{" "}
            <a href={result.humLogsUrl} target="_blank" rel="noopener noreferrer">
              {result.humLogsUrl}
            </a>
          </p>

          <p>HUM melody_ok : <b>{String(result.humSignatureOk)}</b></p>
          {result.humMelodyHash && <p>HUM melody_hash : <b>{result.humMelodyHash}</b></p>}
          {result.humVoicedRatio != null && <p>HUM voiced_ratio : <b>{String(result.humVoicedRatio)}</b></p>}
          {result.humSignatureLen != null && <p>HUM signature_len : <b>{String(result.humSignatureLen)}</b></p>}

          <pre style={{ whiteSpace: "pre-wrap" }}>
            {JSON.stringify(result.humUploadInfo, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
