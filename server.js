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
    // Requêtes sans Origin (curl, server-to-server)
    if (!origin) return cb(null, true);

    if (allowedOrigins.has(origin)) return cb(null, true);

    console.error("❌ CORS blocked origin:", origin);
    return cb(new Error("Not allowed by CORS: " + origin));
  },
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization", "X-Requested-With"],
  // important si un jour tu utilises cookies; sinon laisse false
  credentials: false,
  maxAge: 86400,
};

// IMPORTANT : CORS d'abord
app.use(cors(corsOptions));
app.options("*", cors(corsOptions));

// (optionnel) log toutes les requêtes entrantes (utile Render)
app.use((req, _res, next) => {
  console.log(`➡️ ${req.method} ${req.url} | origin=${req.headers.origin || "none"}`);
  next();
});

// JSON
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// =========================
// Upload config
// =========================
// Limites pour éviter crash / timeouts silencieux
const upload = multer({
  dest: "/tmp",
  limits: {
    fileSize: 15 * 1024 * 1024, // 15MB (ajuste si besoin)
  },
});

const pythonPath = path.join(__dirname, "fingerprint.py");
const API_TOKEN = "3523e792bbced184caa4f51a33a2494a";

// =========================
// Stockage résultats par jobId
// =========================
const resultsByJobId = Object.create(null);

// =========================
// Persistence /tmp
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

// nettoyage jobs vieux (2h)
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

// =========================
// 1) Melody (AUdD)
// =========================
app.post("/melody/upload", upload.single("file"), async (req, res) => {
  console.log("✅ HIT /melody/upload", "origin=", req.headers.origin, "jobId=", req.body?.jobId);

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
      return res.json({
        status: "ok",
        jobId,
        pollUrl: `/melody/result/${jobId}`,
        message: "AUdD OK",
      });
    } catch (err) {
      console.error("❌ AUdD error:", err);
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
      return res.json({ status: "ok", jobId, message: "Python OK" });
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
// 2) Fingerprint generic (inchangé)
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
  console.log("✅ HIT /fingerprint/upload", "origin=", req.headers.origin, "jobId=", req.body?.jobId);

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
// 3) HUM routes
// =========================
app.post("/fingerprint/hum/url", async (req, res) => {
  const { url, jobId } = req.body;
  console.log("✅ HIT /fingerprint/hum/url", "origin=", req.headers.origin, "jobId=", jobId);

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
    logsUrl: `/fingerprint/logs/${jobId}`,
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
  // PROUVE QUE LA ROUTE EST APPELEE
  console.log("✅ HIT /fingerprint/hum/upload", "origin=", req.headers.origin, "jobId=", req.body?.jobId);

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
    logsUrl: `/fingerprint/logs/${jobId}`,
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
// Lancement serveur
// =========================
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Node API running on port ${PORT}`));
