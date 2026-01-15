import express from "express";
import multer from "multer";
import fs from "fs";
import { spawn } from "child_process";
import path from "path";

const app = express();
const upload = multer({ dest: "/tmp" });

// Chemin vers le script Python
const pythonPath = path.join(process.cwd(), "app.py");

// Stockage temporaire de la dernière réponse de Python
let lastPythonResponse = null;

// Route POST où Wix envoie la mélodie
app.post("/melody/upload", upload.single("file"), async (req, res) => {
  try {
    console.log("📥 Audio reçu depuis Wix");

    const filePath = req.file.path;

    // Lancer Python avec le fichier audio
    const python = spawn("python3", [pythonPath, filePath]);

    let data = "";
    let errorData = "";

    python.stdout.on("data", chunk => data += chunk.toString());
    python.stderr.on("data", chunk => errorData += chunk.toString());

    python.on("close", code => {
      fs.unlinkSync(filePath); // supprimer le fichier temporaire

      if (code !== 0) {
        console.error("❌ Erreur Python :", errorData);
        return res.status(500).json({ status: "error", message: errorData });
      }

      try {
        const json = JSON.parse(data);
        lastPythonResponse = json; // stocker la réponse pour GET
        res.json({ status: "ok", message: "Fichier traité, prêt pour fetch" });
      } catch (e) {
        console.error("❌ Erreur parsing JSON :", e.message);
        res.status(500).json({ status: "error", message: e.message });
      }
    });

  } catch (err) {
    console.error("❌ Erreur Node:", err.message);
    res.status(500).json({ status: "error", message: err.message });
  }
});

// Route GET pour que Wix récupère la réponse
app.get("/melody/result", (req, res) => {
  if (!lastPythonResponse) {
    return res.status(404).json({ status: "error", message: "Pas de résultat disponible" });
  }
  res.json({ status: "ok", result: lastPythonResponse });
});

// Lancer le serveur Node
app.listen(3000, () => {
  console.log("🚀 Node API démarrée sur le port 3000");
});
