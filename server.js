import express from "express";
import multer from "multer";
import fs from "fs";
import { spawn } from "child_process";
import path from "path";

const app = express();

/**
 * Render fournit /tmp → OK pour fichiers temporaires
 */
const upload = multer({ dest: "/tmp" });

/**
 * Chemin absolu vers app.py
 */
const pythonPath = path.join(process.cwd(), "app.py");

/**
 * Stockage temporaire du dernier résultat
 * (OK pour MVP / démo — à améliorer plus tard)
 */
let lastPythonResponse = null;

/**
 * =========================
 * POST /melody/upload
 * Wix / React envoie l'audio
 * =========================
 */
app.post("/melody/upload", upload.single("file"), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        status: "error",
        message: "Aucun fichier audio reçu"
      });
    }

    console.log("📥 Audio reçu :", req.file.originalname);

    const filePath = req.file.path;

    /**
     * Lancer Python avec le fichier audio
     */
    const python = spawn("python3", [pythonPath, filePath]);

    let stdoutData = "";
    let stderrData = "";

    python.stdout.on("data", chunk => {
      stdoutData += chunk.toString();
    });

    python.stderr.on("data", chunk => {
      stderrData += chunk.toString();
    });

    python.on("close", code => {
      // Nettoyage fichier temporaire
      fs.unlink(filePath, () => {});

      if (code !== 0) {
        console.error("❌ Python a quitté avec erreur :", stderrData);
        return res.status(500).json({
          status: "error",
          message: "Erreur lors du traitement IA"
        });
      }

      try {
        const result = JSON.parse(stdoutData);
        lastPythonResponse = result;

        res.json({
          status: "ok",
          message: "Audio traité avec succès"
        });

      } catch (err) {
        console.error("❌ JSON invalide retourné par Python :", stdoutData);
        res.status(500).json({
          status: "error",
          message: "Réponse IA invalide"
        });
      }
    });

  } catch (err) {
    console.error("❌ Erreur Node :", err.message);
    res.status(500).json({
      status: "error",
      message: err.message
    });
  }
});

/**
 * =========================
 * GET /melody/result
 * Wix récupère le résultat
 * =========================
 */
app.get("/melody/result", (req, res) => {
  if (!lastPythonResponse) {
    return res.status(404).json({
      status: "error",
      message: "Aucun résultat disponible"
    });
  }

  res.json({
    status: "ok",
    result: lastPythonResponse
  });
});

/**
 * =========================
 * Lancement serveur (OBLIGATOIRE Render)
 * =========================
 */
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Node API démarrée sur le port ${PORT}`);
});
