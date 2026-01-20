import express from "express";
import multer from "multer";
import fs from "fs";
import { spawn } from "child_process";
import path from "path";
import fetch from "node-fetch";
import FormData from "form-data";
import cors from "cors";
import { fileURLToPath } from "url";
import { dirname } from "path";
import { pipeline } from "stream/promises";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();

// =========================
// CORS
// =========================
app.use(
  cors({
    origin: ["https://ia-melodie-1.onrender.com", "http://localhost:5173"],
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
  })
);

// ⚠️ augmente la limite: candidates peut être volumineux
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// =========================
// Upload config
// =========================
const upload = multer({ dest: "/tmp" });
const pythonPath = path.join(__dirname, "fingerprint.py");
const API_TOKEN = "3523e792bbced184caa4f51a33a2494a";

// =========================
// Stockage résultats par jobId
// =========================
const resultsByJobId = Object.create(null);

// =========================
// ✅ Persistence /tmp (anti "JobID inconnu")
// =========================
function jobFilePath(jobId) {
  return `/tmp/fpjob-${jobId}.json`;
}

function saveJob(jobId) {
  try {
    const job = resultsByJobId[jobId];
    if (!job) return;
    fs.writeFileSync(jobFilePath(jobId), JSON.stringify(job), "utf-8");
  } catch (e) {
    console.error("❌ saveJob failed:", e.message);
  }
}

function loadJob(jobId) {
  try {
    const p = jobFilePath(jobId);
    if (!fs.existsSync(p)) return null;
    const raw = fs.readFileSync(p, "utf-8");
    return JSON.parse(raw);
  } catch (e) {
    console.error("❌ loadJob failed:", e.message);
    return null;
  }
}

// (Optionnel) nettoyage jobs vieux (2h)
const JOB_TTL_MS = 2 * 60 * 60 * 1000;
setInterval(() => {
  const now = Date.now();
  for (const [jobId, job] of Object.entries(resultsByJobId)) {
    if (!job?.createdAt) continue;
    if (now - job.createdAt > JOB_TTL_MS) {
      delete resultsByJobId[jobId];
      try {
        const p = jobFilePath(jobId);
        if (fs.existsSync(p)) fs.unlinkSync(p);
      } catch {}
    }
  }
}, 10 * 60 * 1000).unref();

// =========================
// Utils
// =========================
function ensureJob(jobId) {
  if (!resultsByJobId[jobId]) {
    const restored = loadJob(jobId);
    if (restored) {
      resultsByJobId[jobId] = restored;
      return resultsByJobId[jobId];
    }

    resultsByJobId[jobId] = {
      status: "pending",
      result: null,
      error: null,
      logs: [],
      createdAt: Date.now(),
    };
    saveJob(jobId);
  }
  return resultsByJobId[jobId];
}

function pushLog(jobId, line) {
  const job = ensureJob(jobId);
  const msg = `[${new Date().toISOString()}] ${line}`;
  job.logs.push(msg);
  if (job.logs.length > 500) job.logs.shift();
  console.log(`🧾 [${jobId}] ${line}`);
  saveJob(jobId);
}

async function downloadToFile(url, destPath, jobId) {
  pushLog(jobId, `Téléchargement du fichier depuis ${url}`);
  const res = await fetch(url);
  if (!res.ok || !res.body) {
    throw new Error(`Download failed (${res.status})`);
  }
  await pipeline(res.body, fs.createWriteStream(destPath));
  pushLog(jobId, `Téléchargement OK -> ${destPath}`);
}

function runPythonFingerprint(filePath, jobId) {
  return new Promise((resolve, reject) => {
    pushLog(jobId, `Lancement Python: python3 ${pythonPath} ${filePath}`);

    const py = spawn("python3", [pythonPath, filePath], {
      env: { ...process.env },
    });

    let stdout = "";
    let stderrBuffer = "";

    py.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    py.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderrBuffer += text;

      const lines = text.split(/\r?\n/).filter(Boolean);
      for (const l of lines) pushLog(jobId, `PY: ${l}`);
    });

    py.on("error", (err) => {
      pushLog(jobId, `Python spawn error: ${err.message}`);
      reject(err);
    });

    py.on("close", (code) => {
      pushLog(jobId, `Python terminé avec code=${code}`);

      if (code !== 0) {
        const errMsg = `Python error (code=${code}). Stderr(last): ${stderrBuffer.slice(-2000)}`;
        return reject(new Error(errMsg));
      }

      try {
        const parsed = JSON.parse(stdout.trim());
        return resolve(parsed);
      } catch (e) {
        const errMsg = `JSON invalide retourné par Python. stdout(last): ${stdout.slice(-2000)}`;
        return reject(new Error(errMsg));
      }
    });
  });
}

// =========================
// ✅ DTW vectoriel (pour match chroma séquence)
// =========================
function reshapeFlatToFrames(signatureFlat, shape) {
  // shape = [T, 12]
  if (!Array.isArray(shape) || shape.length !== 2) return null;
  const T = Number(shape[0]) || 0;
  const D = Number(shape[1]) || 0;
  if (T <= 0 || D <= 0) return null;
  if (!Array.isArray(signatureFlat) || signatureFlat.length !== T * D) return null;

  const frames = new Array(T);
  for (let t = 0; t < T; t++) {
    const off = t * D;
    const v = new Array(D);
    for (let d = 0; d < D; d++) v[d] = Number(signatureFlat[off + d]) || 0;
    frames[t] = v;
  }
  return frames;
}

function l1DistanceVec(a, b) {
  // a,b length 12, ints 0..127
  const D = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < D; i++) s += Math.abs(a[i] - b[i]);
  return s;
}

function dtwDistanceVec(A, B) {
  const n = A.length;
  const m = B.length;
  if (n === 0 || m === 0) return Number.POSITIVE_INFINITY;

  const INF = 1e18;
  // DP "2 lignes" pour économiser RAM
  let prev = new Float64Array(m + 1);
  let cur = new Float64Array(m + 1);
  for (let j = 0; j <= m; j++) prev[j] = INF;
  prev[0] = 0;

  for (let i = 1; i <= n; i++) {
    cur[0] = INF;
    for (let j = 1; j <= m; j++) {
      const cost = l1DistanceVec(A[i - 1], B[j - 1]);
      const bestPrev = Math.min(prev[j], cur[j - 1], prev[j - 1]);
      cur[j] = cost + bestPrev;
    }
    // swap
    const tmp = prev;
    prev = cur;
    cur = tmp;
  }

  return prev[m];
}

function matchScoreChroma(qFrames, cFrames) {
  const dist = dtwDistanceVec(qFrames, cFrames);

  // dist typique ~ (L * 12 * ~20-60) => ça peut être gros
  // k à tuner: plus grand => scores plus hauts
  const k = 120000; // bon départ; ajuste selon ton dataset
  const score = 1 / (1 + dist / k);
  return Math.max(0, Math.min(1, score));
}

// =========================
// ✅ Melody DTW (ancien) : utile si tu veux fallback mono
// =========================
function dtwDistanceScalar(a, b) {
  const n = a.length;
  const m = b.length;
  const INF = 1e15;

  const dp = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(INF));
  dp[0][0] = 0;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = Math.abs(a[i - 1] - b[j - 1]);
      const bestPrev = Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      dp[i][j] = cost + bestPrev;
    }
  }
  return dp[n][m];
}

function melodyScoreIntervals(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length === 0 || b.length === 0) return 0;

  const dist = dtwDistanceScalar(a, b);
  const k = 2500;
  const score = 1 / (1 + dist / k);
  return Math.max(0, Math.min(1, score));
}

// =========================
// Ping backend
// =========================
app.get("/ping", (req, res) => res.json({ status: "ok", message: "Backend awake" }));

// =========================
// 1️⃣ Routes Melody (AUdD)
// =========================
app.post("/melody/upload", upload.single("file"), async (req, res) => {
  const { jobId } = req.body;
  const backend = req.query.backend || "python";

  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID manquant" });

  const filePath = req.file.path;

  if (backend === "audd") {
    const formData = new FormData();
    formData.append("api_token", API_TOKEN);
    formData.append("file", fs.createReadStream(filePath));
    formData.append("return", "spotify,apple_music");

    try {
      const response = await fetch("https://api.audd.io/", { method: "POST", body: formData });
      const data = await response.json();

      ensureJob(jobId);
      resultsByJobId[jobId].status = "done";
      resultsByJobId[jobId].result = data.result;
      saveJob(jobId);

      fs.unlink(filePath, () => {});
      return res.json({ status: "ok", jobId, message: "Upload reçu, résultat dispo sur /melody/result/:jobId" });
    } catch (err) {
      console.error(err);
      fs.unlink(filePath, () => {});
      return res.status(500).json({ status: "error", message: "AUdD API error" });
    }
  }

  // Backend python (optionnel)
  console.log("📥 Audio reçu (Python) :", req.file.originalname);
  const py = spawn("python3", [pythonPath, filePath]);
  let stdoutData = "";
  let stderrData = "";

  py.stdout.on("data", (chunk) => (stdoutData += chunk.toString()));
  py.stderr.on("data", (chunk) => (stderrData += chunk.toString()));

  py.on("close", (code) => {
    fs.unlink(filePath, () => {});
    if (code !== 0) {
      console.error("❌ Python error :", stderrData);
      return res.status(500).json({ status: "error", message: "Erreur lors du traitement Python" });
    }
    try {
      const parsed = JSON.parse(stdoutData);
      ensureJob(jobId);
      resultsByJobId[jobId].status = "done";
      resultsByJobId[jobId].result = parsed;
      saveJob(jobId);
      return res.json({ status: "ok", jobId, message: "Upload reçu, résultat dispo sur /melody/result/:jobId" });
    } catch {
      console.error("❌ JSON invalide retourné par Python :", stdoutData);
      return res.status(500).json({ status: "error", message: "Réponse Python invalide" });
    }
  });
});

app.get("/melody/result/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID manquant" });

  const job = resultsByJobId[jobId] || loadJob(jobId);
  if (!job || !job.result) return res.status(404).json({ status: "error", message: "Résultat non trouvé" });

  resultsByJobId[jobId] = job;
  return res.json(job.result);
});

// =========================
// 2️⃣ Routes Fingerprint (asynchrones + URL résultat)
// =========================
app.post("/fingerprint/url", async (req, res) => {
  const { url, jobId } = req.body;
  if (!url || !jobId) return res.status(400).json({ status: "error", message: "URL or JobID missing" });

  const job = ensureJob(jobId);

  if (job.status === "done" || job.status === "processing") {
    return res.json({
      status: "ok",
      jobId,
      message: job.status === "done" ? "Déjà calculé" : "Déjà en cours",
      pollUrl: `/fingerprint/${jobId}`,
      resultUrl: `/fingerprint/result/${jobId}`,
    });
  }

  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(jobId);

  pushLog(jobId, "Job fingerprint démarré (URL).");
  const tmpFile = `/tmp/${jobId}.audio`;

  res.json({
    status: "ok",
    jobId,
    message: "Job accepté, traitement en cours",
    pollUrl: `/fingerprint/${jobId}`,
    resultUrl: `/fingerprint/result/${jobId}`,
  });

  (async () => {
    try {
      await downloadToFile(url, tmpFile, jobId);
      const result = await runPythonFingerprint(tmpFile, jobId);

      const j = ensureJob(jobId);
      j.status = "done";
      j.result = result;
      j.error = null;
      saveJob(jobId);
      pushLog(jobId, "Job fingerprint terminé ✅");
    } catch (err) {
      const j = ensureJob(jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(jobId);
      pushLog(jobId, `Job fingerprint échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(tmpFile, () => {});
    }
  })();
});

app.post("/fingerprint/upload", upload.single("file"), async (req, res) => {
  const { jobId } = req.body;
  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const job = ensureJob(jobId);

  if (job.status === "done") {
    fs.unlink(req.file.path, () => {});
    return res.json({
      status: "ok",
      jobId,
      message: "Déjà calculé",
      pollUrl: `/fingerprint/${jobId}`,
      resultUrl: `/fingerprint/result/${jobId}`,
    });
  }

  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(jobId);
  pushLog(jobId, "Job fingerprint démarré (UPLOAD).");

  res.json({
    status: "ok",
    jobId,
    message: "Upload accepté, traitement en cours",
    pollUrl: `/fingerprint/${jobId}`,
    resultUrl: `/fingerprint/result/${jobId}`,
  });

  const filePath = req.file.path;

  (async () => {
    try {
      const result = await runPythonFingerprint(filePath, jobId);
      const j = ensureJob(jobId);
      j.status = "done";
      j.result = result;
      j.error = null;
      saveJob(jobId);
      pushLog(jobId, "Job fingerprint terminé ✅");
    } catch (err) {
      const j = ensureJob(jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(jobId);
      pushLog(jobId, `Job fingerprint échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(filePath, () => {});
    }
  })();
});

app.get("/fingerprint/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const job = resultsByJobId[jobId] || loadJob(jobId);
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });
  resultsByJobId[jobId] = job;

  if (job.status === "done") {
    return res.json({
      status: "done",
      jobId,
      fingerprint: job.result?.fingerprint || null,
      resultUrl: `/fingerprint/result/${jobId}`,
    });
  }

  if (job.status === "error") {
    return res.json({
      status: "error",
      jobId,
      message: job.error || "Erreur inconnue",
      resultUrl: `/fingerprint/result/${jobId}`,
    });
  }

  return res.json({
    status: job.status,
    jobId,
    resultUrl: `/fingerprint/result/${jobId}`,
  });
});

app.get("/fingerprint/result/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const job = resultsByJobId[jobId] || loadJob(jobId);
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });
  resultsByJobId[jobId] = job;

  if (job.status === "done") {
    return res.json({
      status: "done",
      jobId,
      ...job.result,
    });
  }

  if (job.status === "error") {
    return res.status(500).json({
      status: "error",
      jobId,
      message: job.error || "Erreur inconnue",
      logsTail: (job.logs || []).slice(-30),
    });
  }

  return res.status(202).json({
    status: job.status,
    jobId,
    message: "Pas prêt",
  });
});

app.get("/fingerprint/logs/:jobId", (req, res) => {
  const { jobId } = req.params;
  const job = resultsByJobId[jobId] || loadJob(jobId);
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });

  resultsByJobId[jobId] = job;
  return res.json({ status: "ok", jobId, logs: job.logs || [] });
});

// =========================
// 3️⃣ HUM routes (fredonnement / chorale)
//    Maintenant on renvoie match + melody (si dispo)
// =========================
app.post("/fingerprint/hum/url", async (req, res) => {
  const { url, jobId } = req.body;
  if (!url || !jobId) return res.status(400).json({ status: "error", message: "URL or JobID missing" });

  const job = ensureJob(jobId);

  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(jobId);

  pushLog(jobId, "Job HUM démarré (URL).");
  const tmpFile = `/tmp/hum-${jobId}.audio`;

  res.json({
    status: "ok",
    jobId,
    message: "HUM accepté, traitement en cours",
    pollUrl: `/fingerprint/${jobId}`,
    resultUrl: `/fingerprint/result/${jobId}`,
  });

  (async () => {
    try {
      await downloadToFile(url, tmpFile, jobId);
      const result = await runPythonFingerprint(tmpFile, jobId);

      const j = ensureJob(jobId);
      j.status = "done";
      j.result = {
        hum: true,
        match: result.match || null,
        melody: result.melody || null,
        meta: result.meta || null,
      };
      j.error = null;
      saveJob(jobId);
      pushLog(jobId, "Job HUM terminé ✅");
    } catch (err) {
      const j = ensureJob(jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(jobId);
      pushLog(jobId, `Job HUM échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(tmpFile, () => {});
    }
  })();
});

app.post("/fingerprint/hum/upload", upload.single("file"), async (req, res) => {
  const { jobId } = req.body;
  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const job = ensureJob(jobId);

  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(jobId);

  pushLog(jobId, "Job HUM démarré (UPLOAD).");

  res.json({
    status: "ok",
    jobId,
    message: "HUM upload accepté, traitement en cours",
    pollUrl: `/fingerprint/${jobId}`,
    resultUrl: `/fingerprint/result/${jobId}`,
  });

  const filePath = req.file.path;

  (async () => {
    try {
      const result = await runPythonFingerprint(filePath, jobId);

      const j = ensureJob(jobId);
      j.status = "done";
      j.result = {
        hum: true,
        match: result.match || null,
        melody: result.melody || null,
        meta: result.meta || null,
      };
      j.error = null;
      saveJob(jobId);
      pushLog(jobId, "Job HUM terminé ✅");
    } catch (err) {
      const j = ensureJob(jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(jobId);
      pushLog(jobId, `Job HUM échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(filePath, () => {});
    }
  })();
});

// =========================
// 4️⃣ MATCH route (DTW vectoriel sur chroma match.signature)
// Body (recommandé):
// {
//   jobId: "hum-xxx",
//   query: { signature:[..], shape:[T,12] },         // <= query.match.signature de Python
//   candidates: [{ id:"track1", signature:[..], shape:[T,12] }, ...],  // <= stored match.signature
//   topK: 5
// }
//
// Fallback legacy:
// - si pas de shape => on assume melody intervals (scalaire)
// =========================
app.post("/fingerprint/match", async (req, res) => {
  const { jobId, query, candidates, topK } = req.body;

  if (!jobId) return res.status(400).json({ status: "error", message: "jobId missing" });
  if (!query?.signature || !Array.isArray(query.signature)) {
    return res.status(400).json({ status: "error", message: "query.signature missing" });
  }
  if (!Array.isArray(candidates) || candidates.length === 0) {
    return res.status(400).json({ status: "error", message: "candidates missing" });
  }

  const K = Math.max(1, Math.min(20, Number(topK) || 5));
  pushLog(jobId, `MATCH demandé. candidates=${candidates.length}, topK=${K}`);

  const useChromaMatch = Array.isArray(query.shape) && query.shape.length === 2;

  // ---------- MODE CHROMA (polyphonique) ----------
  if (useChromaMatch) {
    const qFrames = reshapeFlatToFrames(query.signature, query.shape);
    if (!qFrames) {
      return res.status(400).json({
        status: "error",
        message: "query.shape/signature invalid (expected flat T*12 with shape [T,12])",
      });
    }

    const scored = [];
    for (const c of candidates) {
      if (!c?.id) continue;
      if (!Array.isArray(c.signature) || !Array.isArray(c.shape)) continue;
      const cFrames = reshapeFlatToFrames(c.signature, c.shape);
      if (!cFrames) continue;

      const score = matchScoreChroma(qFrames, cFrames);
      scored.push({ id: c.id, score });
    }

    scored.sort((a, b) => b.score - a.score);
    const best = scored.slice(0, K);
    const bestScore = best[0]?.score ?? 0;

    return res.json({
      status: "done",
      jobId,
      mode: "chroma_match_v1",
      bestScore,
      bestMatches: best,
    });
  }

  // ---------- MODE MELODY (fallback mono) ----------
  const qSig = query.signature;
  const scored = [];
  for (const c of candidates) {
    if (!c?.id || !Array.isArray(c.signature) || c.signature.length === 0) continue;
    const score = melodyScoreIntervals(qSig, c.signature);
    scored.push({ id: c.id, score });
  }

  scored.sort((a, b) => b.score - a.score);
  const best = scored.slice(0, K);
  const bestScore = best[0]?.score ?? 0;

  return res.json({
    status: "done",
    jobId,
    mode: "melody_intervals_fallback",
    bestScore,
    bestMatches: best,
  });
});

// =========================
// Lancement serveur
// =========================
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Node API running on port ${PORT}`));
