import express from "express";
import multer from "multer";
import fs from "fs";
import { spawn } from "child_process";
import path from "path";
import fetch from "node-fetch";
import FormData from "form-data";
import cors from "cors";

const app = express();

// =========================
// Activer CORS pour le frontend
// =========================
app.use(cors({
  origin: ["https://ia-melodie-1.onrender.com", "http://localhost:5173"],
  methods: ["GET","POST","OPTIONS"],
  allowedHeaders: ["Content-Type"]
}));

// =========================
// Pour parser JSON et form-data simples
// =========================
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// =========================
// Ping backend
// =========================
app.get("/ping", (req, res) => {
  res.json({ status: "ok", message: "Backend awake" });
});

// =========================
// Config upload
// =========================
const upload = multer({ dest: "/tmp" });
const API_TOKEN = "3523e792bbced184caa4f51a33a2494a";
const pythonPath = path.join(process.cwd(), "app.py");

// =========================
// Stockage des résultats par JobID
// =========================
const resultsByJobId = {};

// =========================
// Upload audio et traitement
// =========================
app.post("/melody/upload", upload.single("file"), async (req, res) => {
  const { jobId } = req.body;
  const backend = req.query.backend || "python";

  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });
  if (!jobId) return res.status(400).json({ status: "error", message: "JobID manquant" });

  const filePath = req.file.path;

  if (backend === "audd") {
    // AUdD pipeline
    const formData = new FormData();
    formData.append("api_token", API_TOKEN);
    formData.append("file", fs.createReadStream(filePath));
    formData.append("return", "spotify,apple_music");

    try {
      const response = await fetch("https://api.audd.io/", { method: "POST", body: formData });
      const data = await response.json();

      // Stockage du résultat par JobID
      resultsByJobId[jobId] = data.result;

      fs.unlink(filePath, () => {});
      return res.json({ status: "ok", jobId, message: "Upload reçu, résultat disponible sur /melody/result/:jobId" });
    } catch (err) {
      console.error(err);
      fs.unlink(filePath, () => {});
      return res.status(500).json({ status: "error", message: "AUdD API error" });
    }
  } else {
    // Python pipeline
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
        resultsByJobId[jobId] = parsed; // stocke le résultat par JobID
        return res.json({ status: "ok", jobId, message: "Upload reçu, résultat disponible sur /melody/result/:jobId" });
      } catch (err) {
        console.error("❌ JSON invalide retourné par Python :", stdoutData);
        return res.status(500).json({ status: "error", message: "Réponse Python invalide" });
      }
    });
  }
});

// =========================
// Récupérer résultat par JobID
// =========================
app.get("/melody/result/:jobId", (req, res) => {
  const { jobId } = req.params;

  if (!jobId) return res.status(400).json({ status: "error", message: "JobID manquant" });

  const result = resultsByJobId[jobId];
  if (!result) return res.status(404).json({ status: "error", message: "Résultat non trouvé pour ce JobID" });

  return res.json(result);
});

// =========================
// Lancement serveur
// =========================
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Node API démarrée sur le port ${PORT}`));
