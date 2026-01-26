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

  const pollJob = async (pollJobId, basePath, { interval = 1500, timeout = 120000 } = {}) => {
    const start = Date.now();
    while (true) {
      if (Date.now() - start > timeout) throw new Error("Timeout polling");

      const r = await fetch(`${apiUrl}${basePath}/${pollJobId}`, { cache: "no-store" });
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

    // =========================
    // AUdD (inchangé)
    // =========================
    const fdAudd = new FormData();
    fdAudd.append("file", blob, "recording.webm");
    fdAudd.append("jobId", baseJobId);

    const auddUploadUrl = `${apiUrl}/melody/upload?backend=audd`;

    // =========================
    // Fingerprint local (Python) -> Wix DB match ensuite
    // =========================
    const fpJobId = `${baseJobId}-fp`;
    const fdFp = new FormData();
    fdFp.append("file", blob, "recording.webm");
    fdFp.append("jobId", fpJobId);

    const fpUploadUrl = `${apiUrl}/fingerprint/upload`;

    // =========================
    // QBH (feature/index) en parallèle
    // =========================
    const qbhJobId = `${baseJobId}-qbh`;
    const fdQbh = new FormData();
    fdQbh.append("file", blob, "recording.webm");
    fdQbh.append("jobId", qbhJobId);

    const qbhUploadUrl = `${apiUrl}/qbh/index/upload`;

    console.log("➡️ AUdD upload =>", auddUploadUrl, baseJobId);
    console.log("➡️ FP   upload =>", fpUploadUrl, fpJobId);
    console.log("➡️ QBH  upload =>", qbhUploadUrl, qbhJobId);

    setStatus("🧠 Envoi AUdD + Fingerprint + QBH en parallèle...");

    const [auddSettled, fpSettled, qbhSettled] = await Promise.allSettled([
      fetch(auddUploadUrl, { method: "POST", body: fdAudd }),
      fetch(fpUploadUrl, { method: "POST", body: fdFp }),
      fetch(qbhUploadUrl, { method: "POST", body: fdQbh }),
    ]);

    // -------------------------
    // AUdD result (inchangé)
    // -------------------------
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

    // -------------------------
    // Fingerprint polling + fetch final JSON
    // -------------------------
    let fpOk = false;
    let fpResultUrl = `${apiUrl}/fingerprint/result/${fpJobId}`;
    let fpLogsUrl = `${apiUrl}/fingerprint/logs/${fpJobId}`;
    let fpUploadInfo = null;

    // ce que Wix va utiliser pour matcher DB :
    let fpMelodyHash = null;      // ex: hash stable de la signature
    let fpSignatureOk = false;    // ex: melody_ok
    let fpSignatureLen = null;
    let fpVoicedRatio = null;

    if (fpSettled.status === "fulfilled") {
      const res = fpSettled.value;
      const json = await readJsonSafe(res);
      fpUploadInfo = { ok: res.ok, status: res.status, json };

      if (res.ok) {
        fpOk = true;

        // si backend renvoie resultUrl/logsUrl
        if (json?.resultUrl) fpResultUrl = `${apiUrl}${json.resultUrl}`;
        if (json?.logsUrl) fpLogsUrl = `${apiUrl}${json.logsUrl}`;

        try {
          const pollData = await pollJob(fpJobId, "/fingerprint");
          const finalUrl = pollData?.resultUrl ? `${apiUrl}${pollData.resultUrl}` : fpResultUrl;

          const finalRes = await fetch(finalUrl, { cache: "no-store" });
          const finalJson = await finalRes.json();

          fpResultUrl = finalUrl;

          // 👉 Ici on suppose que fingerprint.py renvoie un bloc "melody"
          //    (comme ton ancien code HUM). Si ton JSON est différent,
          //    adapte juste ces champs.
          const melody = finalJson?.melody || null;
          fpSignatureOk = Boolean(melody?.melody_ok);
          fpMelodyHash = melody?.melody_hash || null;
          fpVoicedRatio = melody?.voiced_ratio ?? null;
          if (Array.isArray(melody?.signature)) fpSignatureLen = melody.signature.length;
        } catch (e) {
          fpOk = false;
          console.error("FP polling/fetch error:", e);
        }
      } else {
        console.error("FP HTTP error:", res.status, json);
      }
    } else {
      console.error("FP error:", fpSettled.reason);
      fpUploadInfo = { error: String(fpSettled.reason) };
    }

    // -------------------------
    // QBH polling + fetch final JSON
    // -------------------------
    let qbhOk = false;
    let qbhResultUrl = `${apiUrl}/qbh/result/${qbhJobId}`;
    let qbhLogsUrl = `${apiUrl}/qbh/logs/${qbhJobId}`;
    let qbhUploadInfo = null;

    if (qbhSettled.status === "fulfilled") {
      const res = qbhSettled.value;
      const json = await readJsonSafe(res);
      qbhUploadInfo = { ok: res.ok, status: res.status, json };

      if (res.ok) {
        qbhOk = true;

        if (json?.resultUrl) qbhResultUrl = `${apiUrl}${json.resultUrl}`;
        if (json?.logsUrl) qbhLogsUrl = `${apiUrl}${json.logsUrl}`;

        try {
          const pollData = await pollJob(qbhJobId, "/qbh");
          const finalUrl = pollData?.resultUrl ? `${apiUrl}${pollData.resultUrl}` : qbhResultUrl;
          qbhResultUrl = finalUrl;
        } catch (e) {
          qbhOk = false;
          console.error("QBH polling error:", e);
        }
      } else {
        console.error("QBH HTTP error:", res.status, json);
      }
    } else {
      console.error("QBH error:", qbhSettled.reason);
      qbhUploadInfo = { error: String(qbhSettled.reason) };
    }

    // -------------------------
    // Status global
    // -------------------------
    const okCount = [auddOk, fpOk, qbhOk].filter(Boolean).length;
    if (okCount === 3) setStatus("✅ AUdD + Fingerprint + QBH terminés");
    else if (okCount >= 1) setStatus(`⚠️ Partiel (${okCount}/3) : voir détails`);
    else setStatus("❌ Tout a échoué");

    const out = {
      jobId: baseJobId,

      // AUdD
      auddUploadUrl,
      auddResultUrl,
      auddUploadInfo,

      // Fingerprint (pour match Wix DB)
      fpJobId,
      fpUploadUrl,
      fpResultUrl,
      fpLogsUrl,
      fpUploadInfo,
      fpSignatureOk,
      fpMelodyHash,
      fpVoicedRatio,
      fpSignatureLen,

      // QBH
      qbhJobId,
      qbhUploadUrl,
      qbhResultUrl,
      qbhLogsUrl,
      qbhUploadInfo,
    };

    setResult(out);

    // -------------------------
    // Retour Wix
    // -------------------------
    if (returnUrl) {
      try {
        const wixUrl = new URL(decodeURIComponent(returnUrl));
        wixUrl.searchParams.set("jobId", baseJobId);

        // AUdD
        if (auddResultUrl) wixUrl.searchParams.set("resultUrl", auddResultUrl);

        // Fingerprint
        wixUrl.searchParams.set("fpJobId", fpJobId);
        wixUrl.searchParams.set("fpResultUrl", fpResultUrl);
        wixUrl.searchParams.set("fpLogsUrl", fpLogsUrl);
        wixUrl.searchParams.set("fpSignatureOk", String(fpSignatureOk));
        if (fpMelodyHash) wixUrl.searchParams.set("fpMelodyHash", fpMelodyHash);
        if (fpVoicedRatio != null) wixUrl.searchParams.set("fpVoicedRatio", String(fpVoicedRatio));
        if (fpSignatureLen != null) wixUrl.searchParams.set("fpSignatureLen", String(fpSignatureLen));

        // QBH
        wixUrl.searchParams.set("qbhJobId", qbhJobId);
        wixUrl.searchParams.set("qbhResultUrl", qbhResultUrl);
        wixUrl.searchParams.set("qbhLogsUrl", qbhLogsUrl);

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

          <hr />
          <h4>AUdD</h4>
          <p><b>Upload :</b> {result.auddUploadUrl}</p>
          {result.auddResultUrl && (
            <p>
              <b>Résultat :</b>{" "}
              <a href={result.auddResultUrl} target="_blank" rel="noopener noreferrer">
                {result.auddResultUrl}
              </a>
            </p>
          )}
          <pre style={{ whiteSpace: "pre-wrap" }}>
            {JSON.stringify(result.auddUploadInfo, null, 2)}
          </pre>

          <hr />
          <h4>Fingerprint (pour DB Wix)</h4>
          <p><b>fpJobId :</b> {result.fpJobId}</p>
          <p><b>Upload :</b> {result.fpUploadUrl}</p>
          <p>
            <b>JSON :</b>{" "}
            <a href={result.fpResultUrl} target="_blank" rel="noopener noreferrer">
              {result.fpResultUrl}
            </a>
          </p>
          <p>
            <b>Logs :</b>{" "}
            <a href={result.fpLogsUrl} target="_blank" rel="noopener noreferrer">
              {result.fpLogsUrl}
            </a>
          </p>
          <p>melody_ok : <b>{String(result.fpSignatureOk)}</b></p>
          {result.fpMelodyHash && <p>melody_hash : <b>{result.fpMelodyHash}</b></p>}
          {result.fpVoicedRatio != null && <p>voiced_ratio : <b>{String(result.fpVoicedRatio)}</b></p>}
          {result.fpSignatureLen != null && <p>signature_len : <b>{String(result.fpSignatureLen)}</b></p>}
          <pre style={{ whiteSpace: "pre-wrap" }}>
            {JSON.stringify(result.fpUploadInfo, null, 2)}
          </pre>

          <hr />
          <h4>QBH</h4>
          <p><b>qbhJobId :</b> {result.qbhJobId}</p>
          <p><b>Upload :</b> {result.qbhUploadUrl}</p>
          <p>
            <b>JSON :</b>{" "}
            <a href={result.qbhResultUrl} target="_blank" rel="noopener noreferrer">
              {result.qbhResultUrl}
            </a>
          </p>
          <p>
            <b>Logs :</b>{" "}
            <a href={result.qbhLogsUrl} target="_blank" rel="noopener noreferrer">
              {result.qbhLogsUrl}
            </a>
          </p>
          <pre style={{ whiteSpace: "pre-wrap" }}>
            {JSON.stringify(result.qbhUploadInfo, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
