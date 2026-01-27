// server.js (UPDATED: stable numba + anti-thread-crash + env forcé sur CHAQUE spawn python)
// ✅ Remplace ton server.js par celui-ci
//
// ✅ Ajouts indispensables (carrefour 3 voies):
// 1) /fingerprint/hum/upload   -> pour Recorder.jsx (chant / hum)
// 2) /qbh/query/extract/upload -> pour extraire la requête QBH sans candidates (comparaison faite par Wix)
// 3) Séparation des jobs par type (fp / qbh / audd) pour éviter collisions
// 4) Retours uniformes: pollUrl/resultUrl/logsUrl
//
// ✅ AJOUTS (bundle):
// - /bundle/:baseJobId -> 1 seule URL qui regroupe les 3 réponses (audd + fp + qbh)
// - logs Render: affiche 1 ligne "🔗 BUNDLE ..." contenant bundleUrl + (optionnel) les 3 urls

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

process.on("uncaughtException", (err) => {
  console.error("💥 uncaughtException:", err);
});
process.on("unhandledRejection", (err) => {
  console.error("💥 unhandledRejection:", err);
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();

// =========================
// CORS (Render Front + Wix + Dev)
// =========================
const allowedOrigins = new Set([
  "https://ia-melodie-1.onrender.com",
  "https://partitionsmanagers.wixstudio.com",
  "http://localhost:5173",
]);

const corsOptions = {
  origin: (origin, cb) => {
    if (!origin) return cb(null, true);
    if (allowedOrigins.has(origin)) return cb(null, true);
    console.error("❌ CORS blocked origin:", origin);
    return cb(new Error("Not allowed by CORS: " + origin));
  },
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization", "X-Requested-With", "Accept"],
  credentials: false,
  maxAge: 86400,
};

app.use(cors(corsOptions));
app.options("*", cors(corsOptions));

app.use((req, _res, next) => {
  console.log(`➡️ ${req.method} ${req.url} | origin=${req.headers.origin || "none"}`);
  next();
});

// JSON
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true }));

// =========================
// Upload config
// =========================
const upload = multer({
  dest: "/tmp",
  limits: {
    fileSize: 25 * 1024 * 1024, // 25MB
  },
});

// =========================
// Python scripts
// =========================
const pythonFingerprintPath = path.join(__dirname, "fingerprint.py");
const pythonQbhPath = path.join(__dirname, "qbh_engine.py");

const API_TOKEN = "3523e792bbced184caa4f51a33a2494a";

// =========================
// SAFE PY ENV (STABLE NUMBA + ANTI THREAD CRASH)
// =========================
// ⚠️ IMPORTANT: on force cet env sur TOUS les spawn python
const NUMBA_CACHE_DIR = "/tmp/numba_cache";
try {
  fs.mkdirSync(NUMBA_CACHE_DIR, { recursive: true });
} catch {}

const SAFE_PY_ENV = {
  ...process.env,

  // ✅ Numba: garder JIT mais contrôler ressources
  NUMBA_NUM_THREADS: "1",
  NUMBA_CACHE_DIR: NUMBA_CACHE_DIR,

  // ✅ limite threads BLAS/OMP (crash sur petits containers)
  OMP_NUM_THREADS: "1",
  OPENBLAS_NUM_THREADS: "1",
  MKL_NUM_THREADS: "1",
  VECLIB_MAXIMUM_THREADS: "1",
  NUMEXPR_NUM_THREADS: "1",

  // ✅ limite fragmentation mémoire
  MALLOC_ARENA_MAX: "2",
};

// ============================================================
// Job store (TYPED) : évite collisions fp/qbh/audd avec même jobId
// ============================================================
const resultsByJobKey = Object.create(null);
function jobKey(type, jobId) {
  return `${type}:${jobId}`;
}
function jobFilePath(type, jobId) {
  return `/tmp/job-${type}-${jobId}.json`;
}
function saveJob(type, jobId) {
  try {
    const key = jobKey(type, jobId);
    const job = resultsByJobKey[key];
    if (!job) return;
    fs.writeFileSync(jobFilePath(type, jobId), JSON.stringify(job), "utf-8");
  } catch (e) {
    console.error("❌ saveJob failed:", e.message);
  }
}
function loadJob(type, jobId) {
  try {
    const p = jobFilePath(type, jobId);
    if (!fs.existsSync(p)) return null;
    const raw = fs.readFileSync(p, "utf-8");
    return JSON.parse(raw);
  } catch (e) {
    console.error("❌ loadJob failed:", e.message);
    return null;
  }
}
function ensureJob(type, jobId) {
  const key = jobKey(type, jobId);
  if (!resultsByJobKey[key]) {
    const restored = loadJob(type, jobId);
    if (restored) {
      resultsByJobKey[key] = restored;
      return resultsByJobKey[key];
    }
    resultsByJobKey[key] = {
      status: "pending",
      result: null,
      error: null,
      logs: [],
      createdAt: Date.now(),
    };
    saveJob(type, jobId);
  }
  return resultsByJobKey[key];
}
function pushLog(type, jobId, line) {
  const job = ensureJob(type, jobId);
  const msg = `[${new Date().toISOString()}] ${line}`;
  job.logs.push(msg);
  if (job.logs.length > 800) job.logs.shift();
  console.log(`🧾 [${type}:${jobId}] ${line}`);
  saveJob(type, jobId);
}

// nettoyage jobs vieux (2h)
const JOB_TTL_MS = 2 * 60 * 60 * 1000;
setInterval(() => {
  const now = Date.now();
  for (const [key, job] of Object.entries(resultsByJobKey)) {
    if (!job?.createdAt) continue;
    if (now - job.createdAt > JOB_TTL_MS) {
      delete resultsByJobKey[key];
      const [type, jobId] = key.split(":");
      try {
        const p = jobFilePath(type, jobId);
        if (fs.existsSync(p)) fs.unlinkSync(p);
      } catch {}
    }
  }
}, 10 * 60 * 1000).unref();

// =========================
// Utils
// =========================
async function downloadToFile(url, destPath, type, jobId) {
  pushLog(type, jobId, `Téléchargement du fichier depuis ${url}`);
  const res = await fetch(url);
  if (!res.ok || !res.body) {
    throw new Error(`Download failed (${res.status})`);
  }
  await pipeline(res.body, fs.createWriteStream(destPath));
  pushLog(type, jobId, `Téléchargement OK -> ${destPath}`);
}

// =========================
// Python runner (fingerprint.py)
// =========================
function runPythonFingerprint(filePath, type, jobId, extraArgs = []) {
  return new Promise((resolve, reject) => {
    pushLog(type, jobId, `Lancement Python: python3 ${pythonFingerprintPath} ${extraArgs.join(" ")} ${filePath}`);

    const py = spawn("python3", [pythonFingerprintPath, ...extraArgs, filePath], {
      env: SAFE_PY_ENV,
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
      for (const l of lines) pushLog(type, jobId, `PY: ${l}`);
    });

    py.on("error", (err) => {
      pushLog(type, jobId, `Python spawn error: ${err.message}`);
      reject(err);
    });

    py.on("close", (code, signal) => {
      pushLog(type, jobId, `Python terminé avec code=${code} signal=${signal || "none"}`);

      if (signal) {
        const errMsg = `Python killed by signal=${signal}. Stderr(last): ${stderrBuffer.slice(-2000)}`;
        return reject(new Error(errMsg));
      }

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
// Python runner (qbh_engine.py via stdin JSON)
// =========================
function runPythonQBH(payload, type, jobId) {
  return new Promise((resolve, reject) => {
    pushLog(type, jobId, `Lancement QBH: python3 ${pythonQbhPath} (mode=${payload?.mode})`);

    const py = spawn("python3", [pythonQbhPath], {
      env: SAFE_PY_ENV,
      stdio: ["pipe", "pipe", "pipe"],
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
      for (const l of lines) pushLog(type, jobId, `QBH: ${l}`);
    });

    py.on("error", (err) => {
      pushLog(type, jobId, `QBH spawn error: ${err.message}`);
      reject(err);
    });

    py.on("close", (code, signal) => {
      pushLog(type, jobId, `QBH terminé avec code=${code} signal=${signal || "none"}`);

      if (signal) {
        const errMsg = `QBH killed by signal=${signal}. Stderr(last): ${stderrBuffer.slice(-2000)}`;
        return reject(new Error(errMsg));
      }

      if (code !== 0) {
        const errMsg = `QBH error (code=${code}). Stderr(last): ${stderrBuffer.slice(-2000)}`;
        return reject(new Error(errMsg));
      }

      try {
        const parsed = JSON.parse(stdout.trim());
        return resolve(parsed);
      } catch (e) {
        const errMsg = `JSON invalide retourné par QBH. stdout(last): ${stdout.slice(-2000)}`;
        return reject(new Error(errMsg));
      }
    });

    try {
      py.stdin.write(JSON.stringify(payload));
      py.stdin.end();
    } catch (e) {
      reject(e);
    }
  });
}

// =========================
// DEBUG route: test CORS sans upload
// =========================
app.get("/debug/echo", (req, res) => {
  res.json({
    ok: true,
    origin: req.headers.origin || null,
    ua: req.headers["user-agent"] || null,
    time: new Date().toISOString(),
  });
});

// =========================
// Ping backend
// =========================
app.get("/ping", (_req, res) => res.json({ status: "ok", message: "Backend awake" }));

// ============================================================
// VOIE A) Fingerprint - URL helpers
// ============================================================
function fpUrls(req, jobId) {
  return {
    pollUrl: absUrl(req, `/fingerprint/${jobId}`),
    resultUrl: absUrl(req, `/fingerprint/result/${jobId}`),
    logsUrl: absUrl(req, `/fingerprint/logs/${jobId}`),
  };
}

// ============================================================
// VOIE B) QBH - URL helpers
// ============================================================
function qbhUrls(req, jobId) {
  return {
    pollUrl: absUrl(req, `/qbh/${jobId}`),
    resultUrl: absUrl(req, `/qbh/result/${jobId}`),
    logsUrl: absUrl(req, `/qbh/logs/${jobId}`),
  };
}

// ============================================================
// VOIE C) AUdD - URL helpers
// ============================================================
function auddUrls(req, baseJobId) {
  return {
    pollUrl: absUrl(req, `/melody/result/${baseJobId}?backend=audd`),
    resultUrl: absUrl(req, `/melody/result/${baseJobId}?backend=audd`),
  };
}

// ============================================================
// BUNDLE (regroupe AUdD + Fingerprint + QBH)
// baseJobId = job-xxxxx-w6cpsb
// fpJobId   = baseJobId + "-fp"
// qbhJobId  = baseJobId + "-qbh"
// auddJobId = baseJobId
// ============================================================
function bundleUrls(req, baseJobId) {
  const fpJobId = `${baseJobId}-fp`;
  const qbhJobId = `${baseJobId}-qbh`;

  return {
    bundleUrl: absUrl(req, `/bundle/${baseJobId}`),
    audd: auddUrls(req, baseJobId),
    fp: fpUrls(req, fpJobId),
    qbh: qbhUrls(req, qbhJobId),
  };
}

// ✅ LOG RENDER: 1 ligne propre pour voir l'URL bundle (+ les 3 sous urls)
function logBundle(req, baseJobId) {
  const u = bundleUrls(req, baseJobId);
  console.log(
    `🔗 BUNDLE base=${baseJobId} bundle=${u.bundleUrl} | audd=${u.audd.resultUrl} | fp=${u.fp.resultUrl} | qbh=${u.qbh.resultUrl}`
  );
}

// ============================================================
// VOIE C) Melody (AUdD) (job typé "audd")
// ============================================================
app.post("/melody/upload", upload.single("file"), async (req, res) => {
  console.log("✅ HIT /melody/upload", "origin=", req.headers.origin, "jobId=", req.body?.jobId);

  const { jobId } = req.body;

  // ✅ IMPORTANT: log immédiat des urls (bundle + sous urls)
  if (jobId) logBundle(jobId, req.headers.origin);

  // ✅ FIX: par défaut c'est audd, pas python
  const backend = req.query.backend || "audd";

  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID manquant" });

  const filePath = req.file.path;

  if (backend === "audd") {
    const formData = new FormData();
    formData.append("api_token", API_TOKEN);
    formData.append("file", fs.createReadStream(filePath));
    formData.append("return", "spotify,apple_music");

    const type = "audd";

    try {
      const response = await fetch("https://api.audd.io/", { method: "POST", body: formData });
      const data = await response.json();

      const job = ensureJob(type, jobId);
      job.status = "done";
      job.result = data.result;
      job.error = null;
      saveJob(type, jobId);

      fs.unlink(filePath, () => {});
      return res.json({
        status: "ok",
        jobId,
        // ✅ FIX: cohérence backend dans les URLs
        pollUrl: `/melody/result/${jobId}?backend=audd`,
        resultUrl: `/melody/result/${jobId}?backend=audd`,
        message: "AUdD OK",
      });
    } catch (err) {
      console.error("❌ AUdD error:", err);
      fs.unlink(filePath, () => {});
      const job = ensureJob(type, jobId);
      job.status = "error";
      job.error = "AUdD API error";
      job.result = null;
      saveJob(type, jobId);
      return res.status(500).json({ status: "error", message: "AUdD API error" });
    }
  }

  fs.unlink(filePath, () => {});
  return res.status(400).json({ status: "error", message: "backend=python disabled here (use /fingerprint/*)" });
});

app.get("/melody/result/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID manquant" });

  const type = "audd";
  const job = resultsByJobKey[jobKey(type, jobId)] || loadJob(type, jobId);
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });

  resultsByJobKey[jobKey(type, jobId)] = job;

  // ✅ FIX: si pas encore prêt → 202 (pour polling)
  if (job.status === "processing" || job.status === "pending") {
    return res.status(202).json({
      status: job.status,
      jobId,
      pollUrl: `/melody/result/${jobId}?backend=audd`,
      resultUrl: `/melody/result/${jobId}?backend=audd`,
      message: "Pas prêt",
    });
  }

  if (job.status === "error") {
    return res.status(500).json({
      status: "error",
      jobId,
      message: job.error || "Erreur AUdD",
    });
  }

  // done
  return res.json(job.result);
});

// ============================================================
// VOIE A) Fingerprint
// ============================================================

app.post("/fingerprint/url", async (req, res) => {
  const { url, jobId } = req.body;
  if (!url || !jobId) return res.status(400).json({ status: "error", message: "URL or JobID missing" });

  const type = "fp";
  const job = ensureJob(type, jobId);

  if (job.status === "done" || job.status === "processing") {
    const urls = fpUrls(jobId);
    return res.json({
      status: "ok",
      jobId,
      message: job.status === "done" ? "Déjà calculé" : "Déjà en cours",
      ...urls,
    });
  }

  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(type, jobId);

  pushLog(type, jobId, "Job fingerprint démarré (URL).");
  const tmpFile = `/tmp/${jobId}.audio`;

  const urls = fpUrls(jobId);
  res.json({
    status: "ok",
    jobId,
    message: "Job accepté, traitement en cours",
    ...urls,
  });

  (async () => {
    try {
      await downloadToFile(url, tmpFile, type, jobId);
      const result = await runPythonFingerprint(tmpFile, type, jobId, ["--mode", "fp"]);

      const j = ensureJob(type, jobId);
      j.status = "done";
      j.result = result;
      j.error = null;
      saveJob(type, jobId);
      pushLog(type, jobId, "Job fingerprint terminé ✅");
    } catch (err) {
      const j = ensureJob(type, jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(type, jobId);
      pushLog(type, jobId, `Job fingerprint échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(tmpFile, () => {});
    }
  })();
});

app.post("/fingerprint/upload", upload.single("file"), async (req, res) => {
  console.log("✅ HIT /fingerprint/upload", "origin=", req.headers.origin, "jobId=", req.body?.jobId);

  const { jobId } = req.body;

  // ✅ optionnel: si tu utilises aussi baseJobId ici, tu peux logBundle(jobIdSansSuffixe)
  // if (jobId) logBundle(jobId, req.headers.origin);

  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const type = "fp";
  const job = ensureJob(type, jobId);

  if (job.status === "done") {
    fs.unlink(req.file.path, () => {});
    const urls = fpUrls(jobId);
    return res.json({ status: "ok", jobId, message: "Déjà calculé", ...urls });
  }

  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(type, jobId);
  pushLog(type, jobId, "Job fingerprint démarré (UPLOAD).");

  const urls = fpUrls(jobId);
  res.json({ status: "ok", jobId, message: "Upload accepté, traitement en cours", ...urls });

  const filePath = req.file.path;

  (async () => {
    try {
      const result = await runPythonFingerprint(filePath, type, jobId, ["--mode", "fp"]);

      const j = ensureJob(type, jobId);
      j.status = "done";
      j.result = result;
      j.error = null;
      saveJob(type, jobId);
      pushLog(type, jobId, "Job fingerprint terminé ✅");
    } catch (err) {
      const j = ensureJob(type, jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(type, jobId);
      pushLog(type, jobId, `Job fingerprint échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(filePath, () => {});
    }
  })();
});

// ✅ ROUTE pour Recorder.jsx (chant/hum)
app.post("/fingerprint/hum/upload", upload.single("file"), async (req, res) => {
  console.log("✅ HIT /fingerprint/hum/upload", "origin=", req.headers.origin, "jobId=", req.body?.jobId);

  const { jobId } = req.body;

  // ✅ IMPORTANT: ici jobId finit par -fp, on log le baseJobId
  if (jobId) {
    const baseJobId = jobId.replace(/-fp$/, "");
    logBundle(baseJobId, req.headers.origin);
  }

  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const type = "fp";
  const job = ensureJob(type, jobId);

  if (job.status === "done") {
    fs.unlink(req.file.path, () => {});
    const urls = fpUrls(jobId);
    return res.json({ status: "ok", jobId, message: "Déjà calculé", ...urls });
  }

  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(type, jobId);
  pushLog(type, jobId, "Job fingerprint HUM démarré (UPLOAD).");

  const urls = fpUrls(jobId);
  res.json({ status: "ok", jobId, message: "Upload HUM accepté, traitement en cours", ...urls });

  const filePath = req.file.path;

  (async () => {
    try {
      const result = await runPythonFingerprint(filePath, type, jobId, ["--mode", "hum"]);

      const j = ensureJob(type, jobId);
      j.status = "done";
      j.result = result;
      j.error = null;
      saveJob(type, jobId);
      pushLog(type, jobId, "Job fingerprint HUM terminé ✅");
    } catch (err) {
      const j = ensureJob(type, jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(type, jobId);
      pushLog(type, jobId, `Job fingerprint HUM échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(filePath, () => {});
    }
  })();
});

app.get("/fingerprint/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const type = "fp";
  const job = resultsByJobKey[jobKey(type, jobId)] || loadJob(type, jobId);
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });
  resultsByJobKey[jobKey(type, jobId)] = job;

  return res.json({ status: job.status, jobId, resultUrl: `/fingerprint/result/${jobId}` });
});

app.get("/fingerprint/result/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const type = "fp";
  const job = resultsByJobKey[jobKey(type, jobId)] || loadJob(type, jobId);
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });
  resultsByJobKey[jobKey(type, jobId)] = job;

  if (job.status === "done") {
    return res.json({ status: "done", jobId, ...job.result });
  }
  if (job.status === "error") {
    return res.status(500).json({
      status: "error",
      jobId,
      message: job.error || "Erreur inconnue",
      logsTail: (job.logs || []).slice(-30),
    });
  }
  return res.status(202).json({ status: job.status, jobId, message: "Pas prêt" });
});

app.get("/fingerprint/logs/:jobId", (req, res) => {
  const { jobId } = req.params;
  const type = "fp";
  const job = resultsByJobKey[jobKey(type, jobId)] || loadJob(type, jobId);
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });

  resultsByJobKey[jobKey(type, jobId)] = job;
  return res.json({ status: "ok", jobId, logs: job.logs || [] });
});

// ============================================================
// VOIE B) QBH routes (index/query + extract)
// ============================================================

function parseCandidates(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  if (typeof raw === "string") {
    try {
      return JSON.parse(raw);
    } catch {
      return [];
    }
  }
  return [];
}

// ----------- QBH INDEX (upload) -----------
app.post("/qbh/index/upload", upload.single("file"), async (req, res) => {
  console.log("✅ HIT /qbh/index/upload", "origin=", req.headers.origin, "jobId=", req.body?.jobId);

  const { jobId } = req.body;
  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const type = "qbh";
  const job = ensureJob(type, jobId);
  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(type, jobId);

  pushLog(type, jobId, "QBH INDEX démarré (UPLOAD).");

  const urls = qbhUrls(jobId);
  res.json({
    status: "ok",
    jobId,
    message: "QBH index accepté",
    ...urls,
  });

  const filePath = req.file.path;

  (async () => {
    try {
      const payload = { mode: "index", audio_path: filePath, sr: 22050, max_seconds: 12 };
      const result = await runPythonQBH(payload, type, jobId);

      const j = ensureJob(type, jobId);
      j.status = "done";
      j.result = result;
      j.error = null;
      saveJob(type, jobId);
      pushLog(type, jobId, "QBH INDEX terminé ✅");
    } catch (err) {
      const j = ensureJob(type, jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(type, jobId);
      pushLog(type, jobId, `QBH INDEX échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(filePath, () => {});
    }
  })();
});

// ----------- QBH INDEX (url) -----------
app.post("/qbh/index/url", async (req, res) => {
  const { url, jobId } = req.body;
  console.log("✅ HIT /qbh/index/url", "origin=", req.headers.origin, "jobId=", jobId);

  if (!url || !jobId) return res.status(400).json({ status: "error", message: "URL or JobID missing" });

  const type = "qbh";
  const job = ensureJob(type, jobId);
  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(type, jobId);

  pushLog(type, jobId, "QBH INDEX démarré (URL).");

  const urls = qbhUrls(jobId);
  res.json({
    status: "ok",
    jobId,
    message: "QBH index accepté",
    ...urls,
  });

  const tmpFile = `/tmp/qbh-index-${jobId}.audio`;

  (async () => {
    try {
      await downloadToFile(url, tmpFile, type, jobId);
      const payload = { mode: "index", audio_path: tmpFile, sr: 22050, max_seconds: 12 };
      const result = await runPythonQBH(payload, type, jobId);

      const j = ensureJob(type, jobId);
      j.status = "done";
      j.result = result;
      j.error = null;
      saveJob(type, jobId);
      pushLog(type, jobId, "QBH INDEX terminé ✅");
    } catch (err) {
      const j = ensureJob(type, jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(type, jobId);
      pushLog(type, jobId, `QBH INDEX échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(tmpFile, () => {});
    }
  })();
});

// ----------- QBH QUERY (upload) -----------
app.post("/qbh/query/upload", upload.single("file"), async (req, res) => {
  console.log("✅ HIT /qbh/query/upload", "origin=", req.headers.origin, "jobId=", req.body?.jobId);

  const { jobId } = req.body;
  const candidates = parseCandidates(req.body?.candidates);

  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });
  if (!candidates?.length) return res.status(400).json({ status: "error", message: "candidates missing/empty" });

  const type = "qbh";
  const job = ensureJob(type, jobId);
  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(type, jobId);

  pushLog(type, jobId, `QBH QUERY démarré (UPLOAD) candidates=${candidates.length}`);

  const urls = qbhUrls(jobId);
  res.json({
    status: "ok",
    jobId,
    message: "QBH query accepté",
    ...urls,
  });

  const filePath = req.file.path;

  (async () => {
    try {
      const payload = {
        mode: "query",
        audio_path: filePath,
        candidates,
        top_k: 10,
        sr: 22050,
        max_seconds: 12,
      };
      const result = await runPythonQBH(payload, type, jobId);

      const j = ensureJob(type, jobId);
      j.status = "done";
      j.result = result;
      j.error = null;
      saveJob(type, jobId);
      pushLog(type, jobId, "QBH QUERY terminé ✅");
    } catch (err) {
      const j = ensureJob(type, jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(type, jobId);
      pushLog(type, jobId, `QBH QUERY échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(filePath, () => {});
    }
  })();
});

// ----------- QBH QUERY (url) -----------
app.post("/qbh/query/url", async (req, res) => {
  const { url, jobId, candidates } = req.body;
  console.log("✅ HIT /qbh/query/url", "origin=", req.headers.origin, "jobId=", jobId);

  const cand = parseCandidates(candidates);

  if (!url || !jobId) return res.status(400).json({ status: "error", message: "URL or JobID missing" });
  if (!cand?.length) return res.status(400).json({ status: "error", message: "candidates missing/empty" });

  const type = "qbh";
  const job = ensureJob(type, jobId);
  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(type, jobId);

  pushLog(type, jobId, `QBH QUERY démarré (URL) candidates=${cand.length}`);

  const urls = qbhUrls(jobId);
  res.json({
    status: "ok",
    jobId,
    message: "QBH query accepté",
    ...urls,
  });

  const tmpFile = `/tmp/qbh-query-${jobId}.audio`;

  (async () => {
    try {
      await downloadToFile(url, tmpFile, type, jobId);

      const payload = {
        mode: "query",
        audio_path: tmpFile,
        candidates: cand,
        top_k: 10,
        sr: 22050,
        max_seconds: 12,
      };
      const result = await runPythonQBH(payload, type, jobId);

      const j = ensureJob(type, jobId);
      j.status = "done";
      j.result = result;
      j.error = null;
      saveJob(type, jobId);
      pushLog(type, jobId, "QBH QUERY terminé ✅");
    } catch (err) {
      const j = ensureJob(type, jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(type, jobId);
      pushLog(type, jobId, `QBH QUERY échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(tmpFile, () => {});
    }
  })();
});

// ✅ NOUVELLE ROUTE: extract QBH query (sans candidates) -> comparaison faite par Wix
app.post("/qbh/query/extract/upload", upload.single("file"), async (req, res) => {
  console.log("✅ HIT /qbh/query/extract/upload", "origin=", req.headers.origin, "jobId=", req.body?.jobId);

  const { jobId } = req.body;

  // ✅ IMPORTANT: ici jobId finit par -qbh, on log le baseJobId
  if (jobId) {
    const baseJobId = jobId.replace(/-qbh$/, "");
    logBundle(baseJobId, req.headers.origin);
  }

  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const type = "qbh";
  const job = ensureJob(type, jobId);
  job.status = "processing";
  job.error = null;
  job.result = null;
  saveJob(type, jobId);

  pushLog(type, jobId, "QBH QUERY EXTRACT démarré (UPLOAD).");

  const urls = qbhUrls(jobId);
  res.json({
    status: "ok",
    jobId,
    message: "QBH query extract accepté",
    ...urls,
  });

  const filePath = req.file.path;

  (async () => {
    try {
      const payload = {
        mode: "extract_query",
        audio_path: filePath,
        sr: 22050,
        max_seconds: 12,
      };
      const result = await runPythonQBH(payload, type, jobId);

      const j = ensureJob(type, jobId);
      j.status = "done";
      j.result = result;
      j.error = null;
      saveJob(type, jobId);
      pushLog(type, jobId, "QBH QUERY EXTRACT terminé ✅");
    } catch (err) {
      const j = ensureJob(type, jobId);
      j.status = "error";
      j.error = err.message || String(err);
      j.result = null;
      saveJob(type, jobId);
      pushLog(type, jobId, `QBH QUERY EXTRACT échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(filePath, () => {});
    }
  })();
});

// ----------- QBH poll/result/logs -----------
app.get("/qbh/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const type = "qbh";
  const job = resultsByJobKey[jobKey(type, jobId)] || loadJob(type, jobId);
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });
  resultsByJobKey[jobKey(type, jobId)] = job;

  return res.json({ status: job.status, jobId, resultUrl: `/qbh/result/${jobId}` });
});

app.get("/qbh/result/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const type = "qbh";
  const job = resultsByJobKey[jobKey(type, jobId)] || loadJob(type, jobId);
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });
  resultsByJobKey[jobKey(type, jobId)] = job;

  if (job.status === "done") {
    return res.json({ status: "done", jobId, ...job.result });
  }
  if (job.status === "error") {
    return res.status(500).json({
      status: "error",
      jobId,
      message: job.error || "Erreur inconnue",
      logsTail: (job.logs || []).slice(-30),
    });
  }
  return res.status(202).json({ status: job.status, jobId, message: "Pas prêt" });
});

app.get("/qbh/logs/:jobId", (req, res) => {
  const { jobId } = req.params;
  const type = "qbh";
  const job = resultsByJobKey[jobKey(type, jobId)] || loadJob(type, jobId);
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });
  resultsByJobKey[jobKey(type, jobId)] = job;
  return res.json({ status: "ok", jobId, logs: job.logs || [] });
});

// ============================================================
// ✅ BUNDLE ROUTE: 1 seule URL qui regroupe les 3 réponses
// GET /bundle/:baseJobId
// - audd jobId = baseJobId
// - fp  jobId = baseJobId + "-fp"
// - qbh jobId = baseJobId + "-qbh"
// ============================================================
app.get("/bundle/:baseJobId", (req, res) => {
  const { baseJobId } = req.params;
  if (!baseJobId) return res.status(400).json({ status: "error", message: "baseJobId missing" });

  const fpJobId = `${baseJobId}-fp`;
  const qbhJobId = `${baseJobId}-qbh`;

  const auddJob = resultsByJobKey[jobKey("audd", baseJobId)] || loadJob("audd", baseJobId);
  const fpJob = resultsByJobKey[jobKey("fp", fpJobId)] || loadJob("fp", fpJobId);
  const qbhJob = resultsByJobKey[jobKey("qbh", qbhJobId)] || loadJob("qbh", qbhJobId);

  const parts = {
    audd: auddJob ? auddJob.status : "missing",
    fingerprint: fpJob ? fpJob.status : "missing",
    qbh: qbhJob ? qbhJob.status : "missing",
  };

  const allDone = parts.audd === "done" && parts.fingerprint === "done" && parts.qbh === "done";
  const anyError = parts.audd === "error" || parts.fingerprint === "error" || parts.qbh === "error";

  const payload = {
    status: allDone ? "done" : anyError ? "error" : "processing",
    baseJobId,
    urls: bundleUrls(req, baseJobId), // ✅ ABSOLUTES
    parts,
    results: {
      audd: auddJob?.status === "done" ? auddJob.result : null,
      fingerprint: fpJob?.status === "done" ? fpJob.result : null,
      qbh: qbhJob?.status === "done" ? qbhJob.result : null,
    },
    errors: {
      audd: auddJob?.status === "error" ? auddJob.error : null,
      fingerprint: fpJob?.status === "error" ? fpJob.error : null,
      qbh: qbhJob?.status === "error" ? qbhJob.error : null,
    },
  };

  if (payload.status === "done") return res.json(payload);
  if (payload.status === "error") return res.status(500).json(payload);
  return res.status(202).json(payload);
});


// =========================
// Lancement serveur
// =========================
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Node API running on port ${PORT}`));
