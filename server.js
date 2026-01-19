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

app.use(express.json());
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
// Structure:
// resultsByJobId[jobId] = {
//   status: "pending" | "processing" | "done" | "error",
//   result: <json> | null,
//   error: string | null,
//   logs: string[],
//   createdAt: number,
// }
const resultsByJobId = Object.create(null);

// =========================
// Utils
// =========================
function ensureJob(jobId) {
  if (!resultsByJobId[jobId]) {
    resultsByJobId[jobId] = {
      status: "pending",
      result: null,
      error: null,
      logs: [],
      createdAt: Date.now(),
    };
  }
  return resultsByJobId[jobId];
}

function pushLog(jobId, line) {
  const job = ensureJob(jobId);
  const msg = `[${new Date().toISOString()}] ${line}`;
  job.logs.push(msg);
  // éviter explosion mémoire
  if (job.logs.length > 500) job.logs.shift();
  console.log(`🧾 [${jobId}] ${line}`);
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
      env: { ...process.env }, // garde environnement Render
    });

    let stdout = "";
    let stderrBuffer = "";

    // stdout: on garde pour JSON final
    py.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    // stderr: logs en continu
    py.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderrBuffer += text;

      // log par lignes
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

      // Important: fingerprint.py doit sortir UN JSON sur stdout
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
// Ping backend
// =========================
app.get("/ping", (req, res) => res.json({ status: "ok", message: "Backend awake" }));

// =========================
// 1️⃣ Routes Melody (déjà existantes) - inchangées
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

      fs.unlink(filePath, () => {});
      return res.json({ status: "ok", jobId, message: "Upload reçu, résultat disponible sur /melody/result/:jobId" });
    } catch (err) {
      console.error(err);
      fs.unlink(filePath, () => {});
      return res.status(500).json({ status: "error", message: "AUdD API error" });
    }
  } else {
    console.log("📥 Audio reçu (Python) :", req.file.originalname);

    const py = spawn("python3", [pythonPath, filePath]);
    let stdoutData = "";
    let stderrData = "";

    py.stdout.on("data", (chunk) => {
      stdoutData += chunk.toString();
    });
    py.stderr.on("data", (chunk) => {
      stderrData += chunk.toString();
    });

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

        return res.json({ status: "ok", jobId, message: "Upload reçu, résultat disponible sur /melody/result/:jobId" });
      } catch (err) {
        console.error("❌ JSON invalide retourné par Python :", stdoutData);
        return res.status(500).json({ status: "error", message: "Réponse Python invalide" });
      }
    });
  }
});

app.get("/melody/result/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID manquant" });

  const job = resultsByJobId[jobId];
  if (!job || !job.result) return res.status(404).json({ status: "error", message: "Résultat non trouvé pour ce JobID" });

  return res.json(job.result);
});

// =========================
// 2️⃣ Routes Fingerprint (asynchrones + URL résultat)
// =========================

// POST url -> accepte le job, lance traitement en arrière-plan (dans le même process), retourne immédiatement
app.post("/fingerprint/url", async (req, res) => {
  const { url, jobId } = req.body;
  if (!url || !jobId) return res.status(400).json({ status: "error", message: "URL or JobID missing" });

  const job = ensureJob(jobId);

  // si déjà fait, on renvoie direct
  if (job.status === "done") {
    return res.json({
      status: "ok",
      jobId,
      message: "Déjà calculé",
      pollUrl: `/fingerprint/${jobId}`,
      resultUrl: `/fingerprint/result/${jobId}`,
    });
  }

  // si en cours, on renvoie direct
  if (job.status === "processing") {
    return res.json({
      status: "ok",
      jobId,
      message: "Déjà en cours",
      pollUrl: `/fingerprint/${jobId}`,
      resultUrl: `/fingerprint/result/${jobId}`,
    });
  }

  job.status = "processing";
  job.error = null;
  job.result = null;
  pushLog(jobId, "Job fingerprint démarré (URL).");

  const tmpFile = `/tmp/${jobId}.audio`;

  // IMPORTANT: on répond tout de suite (Wix va poll)
  res.json({
    status: "ok",
    jobId,
    message: "Job accepté, traitement en cours",
    pollUrl: `/fingerprint/${jobId}`,
    resultUrl: `/fingerprint/result/${jobId}`,
  });

  // Traitement async après réponse
  (async () => {
    try {
      await downloadToFile(url, tmpFile, jobId);
      const result = await runPythonFingerprint(tmpFile, jobId);

      const j = ensureJob(jobId);
      j.status = "done";
      j.result = result;
      pushLog(jobId, "Job fingerprint terminé ✅");
    } catch (err) {
      const j = ensureJob(jobId);
      j.status = "error";
      j.error = err.message || String(err);
      pushLog(jobId, `Job fingerprint échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(tmpFile, () => {});
    }
  })();
});

// (Optionnel) upload direct fichier multipart -> pareil, asynchrone
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
      pushLog(jobId, "Job fingerprint terminé ✅");
    } catch (err) {
      const j = ensureJob(jobId);
      j.status = "error";
      j.error = err.message || String(err);
      pushLog(jobId, `Job fingerprint échoué ❌ : ${j.error}`);
    } finally {
      fs.unlink(filePath, () => {});
    }
  })();
});

// Polling status (Wix l'utilise)
app.get("/fingerprint/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const job = resultsByJobId[jobId];
  if (!job) {
    return res.status(404).json({ status: "error", message: "JobID inconnu" });
  }

  // IMPORTANT: ton Wix attend {status:"done", fingerprint: ...} éventuellement.
  // Ici, on renvoie aussi resultUrl pour qu’il fetch le JSON complet.
  if (job.status === "done") {
    return res.json({
      status: "done",
      jobId,
      // fingerprint "court" pour affichage rapide
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
    status: job.status, // pending/processing
    jobId,
    resultUrl: `/fingerprint/result/${jobId}`,
  });
});

// URL dédiée: JSON final fetchable
app.get("/fingerprint/result/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const job = resultsByJobId[jobId];
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });

  if (job.status === "done") {
    return res.json({
      status: "done",
      jobId,
      ...job.result, // contient fingerprint + meta
    });
  }

  if (job.status === "error") {
    return res.status(500).json({
      status: "error",
      jobId,
      message: job.error || "Erreur inconnue",
      logsTail: job.logs.slice(-30),
    });
  }

  return res.status(202).json({
    status: job.status,
    jobId,
    message: "Pas prêt",
  });
});

// (Optionnel) logs consultables (debug)
app.get("/fingerprint/logs/:jobId", (req, res) => {
  const { jobId } = req.params;
  const job = resultsByJobId[jobId];
  if (!job) return res.status(404).json({ status: "error", message: "JobID inconnu" });
  return res.json({ status: "ok", jobId, logs: job.logs });
});

// =========================
// Lancement serveur
// =========================
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Node API running on port ${PORT}`));
