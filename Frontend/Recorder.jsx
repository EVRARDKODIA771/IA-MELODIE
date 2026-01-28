import { useRef, useState, useEffect } from "react";
import "./Recorder.css";

const backendUrl = import.meta.env.VITE_BACKEND_URL;
const apiUrl = backendUrl || "https://ia-melodie.onrender.com";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ✅ 7 secondes, comme demandé
const RECORD_MS = 7000;

// ✅ helper: jobIds dédiés (pas de collision)
const makeJobIds = (baseJobId) => ({
  base: baseJobId,
  audd: baseJobId,        // AUdD garde jobId base
  fp: `${baseJobId}-fp`,  // fingerprint
  qbh: `${baseJobId}-qbh` // qbh
});

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

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // ✅ mimeType robuste (certains navigateurs refusent audio/webm)
      let options = {};
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        options = { mimeType: "audio/webm;codecs=opus" };
      } else if (MediaRecorder.isTypeSupported("audio/webm")) {
        options = { mimeType: "audio/webm" };
      }

      const recorder = new MediaRecorder(stream, options);

      mediaRecorderRef.current = recorder;
      chunksRef.current = [];
      setResult(null);

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        setStatus("🧠 Enregistrement terminé, envoi au serveur...");
        const ids = makeJobIds(jobIdFromWix);
        sendAudio(blob, ids).catch((e) => {
          console.error(e);
          setStatus("❌ Erreur pendant l'envoi");
        });
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
      }, RECORD_MS);
    } catch (e) {
      console.error("getUserMedia error:", e);
      setStatus("❌ Micro refusé / indisponible");
    }
  };

  const togglePause = () => {
    if (!mediaRecorderRef.current) return;

    if (mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.pause();
      setIsPaused(true);
      setStatus("⏸️ En pause");
      clearInterval(timerRef.current);
    } else if (mediaRecorderRef.current.state === "paused") {
      mediaRecorderRef.current.resume();
      setIsPaused(false);
      setStatus("🎶 Enregistrement en cours...");
      timerRef.current = setInterval(() => setTime((t) => t + 1), 1000);
    }
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current) return;
    try {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    } catch {}
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

  // ============================================================
  // Envoi en parallèle: AUdD + Fingerprint(HUM) + QBH(EXTRACT_QUERY)
  // + Retour vers Wix avec bundleUrl (/bundle/:baseJobId)
  // ============================================================
  const sendAudio = async (blob, ids) => {
    if (!blob || !ids?.base) return;

    // =========================
    // AUdD (inchangé)
    // =========================
    const fdAudd = new FormData();
    fdAudd.append("file", blob, "recording.webm");
    fdAudd.append("jobId", ids.audd);

    const auddUploadUrl = `${apiUrl}/melody/upload?backend=audd`;

    // =========================
    // Fingerprint HUM
    // =========================
    const fdFp = new FormData();
    fdFp.append("file", blob, "recording.webm");
    fdFp.append("jobId", ids.fp);

    const fpUploadUrl = `${apiUrl}/fingerprint/hum/upload`;

    // =========================
    // QBH extract query
    // =========================
    const fdQbh = new FormData();
    fdQbh.append("file", blob, "recording.webm");
    fdQbh.append("jobId", ids.qbh);

    const qbhUploadUrl = `${apiUrl}/qbh/query/extract/upload`;

    console.log("➡️ AUdD upload =>", auddUploadUrl, ids.audd);
    console.log("➡️ FP(HUM) upload =>", fpUploadUrl, ids.fp);
    console.log("➡️ QBH(EXTRACT) upload =>", qbhUploadUrl, ids.qbh);

    setStatus("🧠 Envoi AUdD + Fingerprint(HUM) + QBH(EXTRACT) en parallèle...");

    const [auddSettled, fpSettled, qbhSettled] = await Promise.allSettled([
      fetch(auddUploadUrl, { method: "POST", body: fdAudd }),
      fetch(fpUploadUrl, { method: "POST", body: fdFp }),
      fetch(qbhUploadUrl, { method: "POST", body: fdQbh }),
    ]);

    // -------------------------
    // AUdD result (comme avant)
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
        auddResultUrl = `${apiUrl}/melody/result/${ids.audd}`;
      } else {
        console.error("AUdD HTTP error:", res.status, json);
      }
    } else {
      console.error("AUdD error:", auddSettled.reason);
      auddUploadInfo = { error: String(auddSettled.reason) };
    }

    // -------------------------
    // Fingerprint polling + fetch final JSON (comme avant)
    // -------------------------
    let fpOk = false;
    let fpResultUrl = `${apiUrl}/fingerprint/result/${ids.fp}`;
    let fpLogsUrl = `${apiUrl}/fingerprint/logs/${ids.fp}`;
    let fpUploadInfo = null;

    let fpSignatureOk = false;
    let fpMelodyHash = null;
    let fpVoicedRatio = null;
    let fpSignatureLen = null;

    if (fpSettled.status === "fulfilled") {
      const res = fpSettled.value;
      const json = await readJsonSafe(res);
      fpUploadInfo = { ok: res.ok, status: res.status, json };

      if (res.ok) {
        fpOk = true;

        // Important: ces champs viennent parfois déjà absolus,
        // donc on ne préfixe QUE si c'est relatif.
        if (json?.resultUrl) {
          fpResultUrl = json.resultUrl.startsWith("http") ? json.resultUrl : `${apiUrl}${json.resultUrl}`;
        }
        if (json?.logsUrl) {
          fpLogsUrl = json.logsUrl.startsWith("http") ? json.logsUrl : `${apiUrl}${json.logsUrl}`;
        }

        try {
          const pollData = await pollJob(ids.fp, "/fingerprint");
          const finalUrl = pollData?.resultUrl
            ? (pollData.resultUrl.startsWith("http") ? pollData.resultUrl : `${apiUrl}${pollData.resultUrl}`)
            : fpResultUrl;

          const finalRes = await fetch(finalUrl, { cache: "no-store" });
          const finalJson = await finalRes.json();

          fpResultUrl = finalUrl;

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
    // QBH polling + fetch final JSON (comme avant)
    // -------------------------
    let qbhOk = false;
    let qbhResultUrl = `${apiUrl}/qbh/result/${ids.qbh}`;
    let qbhLogsUrl = `${apiUrl}/qbh/logs/${ids.qbh}`;
    let qbhUploadInfo = null;

    let qbhQuery = null;
    let qbhQueryLen = null;
    let qbhMeta = null;

    if (qbhSettled.status === "fulfilled") {
      const res = qbhSettled.value;
      const json = await readJsonSafe(res);
      qbhUploadInfo = { ok: res.ok, status: res.status, json };

      if (res.ok) {
        qbhOk = true;

        if (json?.resultUrl) {
          qbhResultUrl = json.resultUrl.startsWith("http") ? json.resultUrl : `${apiUrl}${json.resultUrl}`;
        }
        if (json?.logsUrl) {
          qbhLogsUrl = json.logsUrl.startsWith("http") ? json.logsUrl : `${apiUrl}${json.logsUrl}`;
        }

        try {
          const pollData = await pollJob(ids.qbh, "/qbh");
          const finalUrl = pollData?.resultUrl
            ? (pollData.resultUrl.startsWith("http") ? pollData.resultUrl : `${apiUrl}${pollData.resultUrl}`)
            : qbhResultUrl;

          qbhResultUrl = finalUrl;

          const finalRes = await fetch(finalUrl, { cache: "no-store" });
          const finalJson = await finalRes.json();

          const query = finalJson?.query ?? finalJson?.q ?? null;
          qbhQuery = query;
          qbhQueryLen = Array.isArray(query) ? query.length : null;
          qbhMeta = finalJson?.meta ?? null;
        } catch (e) {
          qbhOk = false;
          console.error("QBH polling/fetch error:", e);
        }
      } else {
        console.error("QBH HTTP error:", res.status, json);
      }
    } else {
      console.error("QBH error:", qbhSettled.reason);
      qbhUploadInfo = { error: String(qbhSettled.reason) };
    }

    // -------------------------
    // ✅ URL BUNDLE UNIQUE (3 réponses fusionnées par server.js)
    // -------------------------
    const bundleUrl = `${apiUrl}/bundle/${ids.base}`;

    // -------------------------
    // Status global
    // -------------------------
    const okCount = [auddOk, fpOk, qbhOk].filter(Boolean).length;
    if (okCount === 3) setStatus("✅ AUdD + Fingerprint(HUM) + QBH(EXTRACT) terminés");
    else if (okCount >= 1) setStatus(`⚠️ Partiel (${okCount}/3) : voir détails`);
    else setStatus("❌ Tout a échoué");

    const out = {
      jobId: ids.base,

      // ✅ bundle unique
      bundleUrl,

      // AUdD
      auddUploadUrl,
      auddResultUrl,
      auddUploadInfo,

      // Fingerprint (match DB Wix)
      fpJobId: ids.fp,
      fpUploadUrl,
      fpResultUrl,
      fpLogsUrl,
      fpUploadInfo,
      fpSignatureOk,
      fpMelodyHash,
      fpVoicedRatio,
      fpSignatureLen,

      // QBH (extract query)
      qbhJobId: ids.qbh,
      qbhUploadUrl,
      qbhResultUrl,
      qbhLogsUrl,
      qbhUploadInfo,
      qbhQuery,
      qbhQueryLen,
      qbhMeta,
    };

    setResult(out);

    // -------------------------
    // ✅ Retour Wix (returnUrl) : on met bundleUrl (et on garde l'ancien pour compat)
    // -------------------------
    if (returnUrl) {
      try {
        const wixUrl = new URL(decodeURIComponent(returnUrl));
        wixUrl.searchParams.set("jobId", ids.base);

        // ✅ NOUVEAU : Wix doit fetcher cette URL pour obtenir les 3 résultats
        wixUrl.searchParams.set("bundleUrl", bundleUrl);

        // (Compat ancien frontend)
        if (auddResultUrl) wixUrl.searchParams.set("resultUrl", auddResultUrl);

        // On peut garder fp/qbh séparés aussi (debug/compat)
        wixUrl.searchParams.set("fpJobId", ids.fp);
        wixUrl.searchParams.set("fpResultUrl", fpResultUrl);
        wixUrl.searchParams.set("fpLogsUrl", fpLogsUrl);
        wixUrl.searchParams.set("fpSignatureOk", String(fpSignatureOk));
        if (fpMelodyHash) wixUrl.searchParams.set("fpMelodyHash", fpMelodyHash);
        if (fpVoicedRatio != null) wixUrl.searchParams.set("fpVoicedRatio", String(fpVoicedRatio));
        if (fpSignatureLen != null) wixUrl.searchParams.set("fpSignatureLen", String(fpSignatureLen));

        wixUrl.searchParams.set("qbhJobId", ids.qbh);
        wixUrl.searchParams.set("qbhResultUrl", qbhResultUrl);
        wixUrl.searchParams.set("qbhLogsUrl", qbhLogsUrl);
        if (qbhQueryLen != null) wixUrl.searchParams.set("qbhQueryLen", String(qbhQueryLen));

        window.location.href = wixUrl.toString();
      } catch (err) {
        console.error("Erreur parsing returnUrl Wix :", err);
      }
    } else {
      // fallback debug
      console.log("bundleUrl:", bundleUrl);
    }
  };

  return (
    <div className="recorder-container">
      <div className="title">PARTITION MANAGER</div>
      <div className="subtitle">Chantez ou fredonnez une musique (7s)</div>

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
          <p>
            <b>JobID :</b> {result.jobId}
          </p>

          <hr />
          <h4>BUNDLE (3 résultats fusionnés)</h4>
          <p>
            <b>JSON :</b>{" "}
            <a href={result.bundleUrl} target="_blank" rel="noopener noreferrer">
              {result.bundleUrl}
            </a>
          </p>

          <hr />
          <h4>AUdD</h4>
          <p>
            <b>Upload :</b> {result.auddUploadUrl}
          </p>
          {result.auddResultUrl && (
            <p>
              <b>Résultat :</b>{" "}
              <a href={result.auddResultUrl} target="_blank" rel="noopener noreferrer">
                {result.auddResultUrl}
              </a>
            </p>
          )}
          <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(result.auddUploadInfo, null, 2)}</pre>

          <hr />
          <h4>Fingerprint (HUM → DB Wix)</h4>
          <p>
            <b>fpJobId :</b> {result.fpJobId}
          </p>
          <p>
            <b>Upload :</b> {result.fpUploadUrl}
          </p>
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
          <p>
            melody_ok : <b>{String(result.fpSignatureOk)}</b>
          </p>
          {result.fpMelodyHash && (
            <p>
              melody_hash : <b>{result.fpMelodyHash}</b>
            </p>
          )}
          {result.fpVoicedRatio != null && (
            <p>
              voiced_ratio : <b>{String(result.fpVoicedRatio)}</b>
            </p>
          )}
          {result.fpSignatureLen != null && (
            <p>
              signature_len : <b>{String(result.fpSignatureLen)}</b>
            </p>
          )}
          <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(result.fpUploadInfo, null, 2)}</pre>

          <hr />
          <h4>QBH (EXTRACT query → comparaison côté Wix)</h4>
          <p>
            <b>qbhJobId :</b> {result.qbhJobId}
          </p>
          <p>
            <b>Upload :</b> {result.qbhUploadUrl}
          </p>
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
          {result.qbhQueryLen != null && (
            <p>
              query_len : <b>{String(result.qbhQueryLen)}</b>
            </p>
          )}
          <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(result.qbhUploadInfo, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
