import express from "express";
import multer from "multer";
import fs from "fs";
import { spawn } from "child_process";
import path from "path";
import fetch from "node-fetch";
import FormData from "form-data";

const app = express();

/**
 * =========================
 * Ping (réveil backend)
 * =========================
 */
app.get("/ping", (req, res) => {
  res.json({ status: "ok", message: "Backend awake" });
});

/**
 * =========================
 * Configuration upload
 * =========================
 */
const upload = multer({ dest: "/tmp" });
const API_TOKEN = "3523e792bbced184caa4f51a33a2494a";
const pythonPath = path.join(process.cwd(), "app.py");

/**
 * Stockage des résultats
 */
let lastPythonResult = null;
let lastAuddResult = null;

/**
 * =========================
 * Endpoint unique upload
 * pipeline=python|audd
 * =========================
 */
app.post("/melody/upload", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ status: "error", message: "No file" });

  const backend = req.query.backend || "python"; // par défaut Python
  const filePath = req.file.path;

  if (backend === "audd") {
    // Pipeline AUdD
    const formData = new FormData();
    formData.append("api_token", API_TOKEN);
    formData.append("file", fs.createReadStream(filePath));
    formData.append("return", "spotify,apple_music");

    try {
      const response = await fetch("https://api.audd.io/", { method: "POST", body: formData });
      const data = await response.json();
      lastAuddResult = data.result;
      fs.unlink(filePath, () => {});
      return res.json({ status: "ok", result: lastAuddResult });
    } catch (err) {
      console.error(err);
      fs.unlink(filePath, () => {});
      return res.status(500).json({ status: "error", message: "AUdD API error" });
    }
  } else {
    // Pipeline Python
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
        lastPythonResult = JSON.parse(stdoutData);
        return res.json({ status: "ok", result: lastPythonResult });
      } catch (err) {
        console.error("❌ JSON invalide retourné par Python :", stdoutData);
        return res.status(500).json({ status: "error", message: "Réponse Python invalide" });
      }
    });
  }
});

/**
 * =========================
 * Récupérer le dernier résultat complet
 * ?backend=python|audd
 * =========================
 */
app.get("/melody/result", (req, res) => {
  const backend = req.query.backend || "python";
  if (backend === "audd") {
    if (!lastAuddResult) return res.status(404).json({ status: "error", message: "Aucun résultat AUdD disponible" });
    return res.json(lastAuddResult);
  } else {
    if (!lastPythonResult) return res.status(404).json({ status: "error", message: "Aucun résultat Python disponible" });
    return res.json(lastPythonResult);
  }
});

/**
 * =========================
 * Endpoints legacy Wix (Python uniquement)
 * =========================
 */
app.get("/result/lyrics", (req, res) => res.json(lastPythonResult?.lyrics || null));
app.get("/result/global", (req, res) => res.json(lastPythonResult?.global_match || null));
app.get("/result/wix", (req, res) => res.json(lastPythonResult?.wix_match || null));

/**
 * =========================
 * Lancement serveur
 * =========================
 */
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Node API démarrée sur le port ${PORT}`));
