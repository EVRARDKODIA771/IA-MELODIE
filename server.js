import express from "express";
import multer from "multer";
import fs from "fs";
import { spawn } from "child_process";
import path from "path";

const app = express();

/**
 * =========================
 * Configuration upload (Render)
 * =========================
 */
const upload = multer({ dest: "/tmp" });

/**
 * Chemin absolu vers app.py
 */
const pythonPath = path.join(process.cwd(), "app.py");

/**
 * Stockage temporaire du dernier résultat Python
 * (OK pour MVP / démo)
 */
let lastResult = null;

/**
 * =========================
 * GET /ping
 * 👉 Réveille le backend Render
 * =========================
 */
app.get("/ping", (req, res) => {
  res.json({
    status: "ok",
    message: "Backend awake"
  });
});

/**
 * =========================
 * POST /melody/upload
 * 👉 Wix / React envoie l'audio
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
        console.error("❌ Python error :", stderrData);
        return res.status(500).json({
          status: "error",
          message: "Erreur lors du traitement Python"
        });
      }

      try {
        /**
         * Résultat Python attendu :
         * {
         *   lyrics: "...",
         *   global_match: {...},
         *   wix_match: {...}
         * }
         */
        lastResult = JSON.parse(stdoutData);

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
 * 👉 Résultat complet (debug / legacy)
 * =========================
 */
app.get("/melody/result", (req, res) => {
  if (!lastResult) {
    return res.status(404).json({
      status: "error",
      message: "Aucun résultat disponible"
    });
  }

  res.json({
    status: "ok",
    result: lastResult
  });
});

/**
 * =========================
 * GET /result/lyrics
 * 👉 Paroles reconnues (toutes langues)
 * =========================
 */
app.get("/result/lyrics", (req, res) => {
  res.json(lastResult?.lyrics || null);
});

/**
 * =========================
 * GET /result/global
 * 👉 Shazam-like (YouTube / Spotify / Web)
 * =========================
 */
app.get("/result/global", (req, res) => {
  res.json(lastResult?.global_match || null);
});

/**
 * =========================
 * GET /result/wix
 * 👉 Match avec ta BDD Wix
 * =========================
 */
app.get("/result/wix", (req, res) => {
  res.json(lastResult?.wix_match || null);
});

/**
 * =========================
 * Lancement serveur (Render)
 * =========================
 */
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Node API démarrée sur le port ${PORT}`);
});
