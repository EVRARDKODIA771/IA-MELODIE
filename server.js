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

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();

// =========================
// CORS
// =========================
app.use(cors({
  origin: ["https://ia-melodie-1.onrender.com", "http://localhost:5173"],
  methods: ["GET","POST","OPTIONS"],
  allowedHeaders: ["Content-Type"]
}));

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
const resultsByJobId = {};

// =========================
// Ping backend
// =========================
app.get("/ping", (req, res) => res.json({ status: "ok", message: "Backend awake" }));

// =========================
// 1️⃣ Routes Melody (déjà existantes)
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

      resultsByJobId[jobId] = data.result;
      fs.unlink(filePath, () => {});
      return res.json({ status: "ok", jobId, message: "Upload reçu, résultat disponible sur /melody/result/:jobId" });
    } catch (err) {
      console.error(err);
      fs.unlink(filePath, () => {});
      return res.status(500).json({ status: "error", message: "AUdD API error" });
    }
  } else {
    console.log("📥 Audio reçu (Python) :", req.file.originalname);

    const python = spawn("python3", [pythonPath, filePath]);
    let stdoutData = "";
    let stderrData = "";

    python.stdout.on("data", chunk => { stdoutData += chunk.toString(); });
    python.stderr.on("data", chunk => { stderrData += chunk.toString(); });

    python.on("close", code => {
      fs.unlink(filePath, () => {});

      if (code !== 0) {
        console.error("❌ Python error :", stderrData);
        return res.status(500).json({ status: "error", message: "Erreur lors du traitement Python" });
      }

      try {
        const parsed = JSON.parse(stdoutData);
        resultsByJobId[jobId] = parsed;
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

  const result = resultsByJobId[jobId];
  if (!result) return res.status(404).json({ status: "error", message: "Résultat non trouvé pour ce JobID" });

  return res.json(result);
});

// =========================
// 2️⃣ Routes Fingerprint (nouveau)
// =========================
app.post("/fingerprint/upload", upload.single("file"), async (req, res) => {
  const { jobId } = req.body;
  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  try {
    const fingerprint = await generateFingerprint(req.file.path);
    resultsByJobId[jobId] = fingerprint;

    fs.unlink(req.file.path, () => {});
    return res.json({ status: "ok", jobId, fingerprint });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ status: "error", message: "Fingerprint failed" });
  }
});

app.post("/fingerprint/url", async (req, res) => {
  const { url, jobId } = req.body;
  if (!url || !jobId) return res.status(400).json({ status: "error", message: "URL or JobID missing" });

  const tmpFile = `/tmp/${jobId}.mp3`;
  try {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    fs.writeFileSync(tmpFile, Buffer.from(buffer));

    const fingerprint = await generateFingerprint(tmpFile);
    resultsByJobId[jobId] = fingerprint;

    fs.unlink(tmpFile, () => {});
    return res.json({ status: "ok", jobId, fingerprint });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ status: "error", message: "Failed to download or fingerprint" });
  }
});

app.get("/fingerprint/:jobId", (req, res) => {
  const { jobId } = req.params;
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID missing" });

  const fingerprint = resultsByJobId[jobId];
  if (!fingerprint) return res.status(404).json({ status: "error", message: "Fingerprint not found" });

  return res.json({ status: "ok", jobId, fingerprint });
});

// =========================
// Générer empreinte via Python
// =========================
function generateFingerprint(filePath) {
  return new Promise((resolve, reject) => {
    const python = spawn("python3", [pythonPath, filePath]);
    let stdout = "";
    let stderr = "";

    python.stdout.on("data", chunk => { stdout += chunk.toString(); });
    python.stderr.on("data", chunk => { stderr += chunk.toString(); });

    python.on("close", code => {
      if (code !== 0) return reject(stderr);
      resolve(stdout.trim());
    });
  });
}

// =========================
// Lancement serveur
// =========================
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Node API running on port ${PORT}`));
