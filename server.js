import express from "express";
import multer from "multer";
import fs from "fs";
import axios from "axios";
import FormData from "form-data";

const app = express();
const upload = multer({ dest: "/tmp" });

// URL du moteur Python (engine)
const PYTHON_ENGINE_URL = process.env.PYTHON_ENGINE_URL || "http://localhost:8000";

app.post("/melody/fingerprint", upload.single("file"), async (req, res) => {
  try {
    console.log("📥 Audio reçu depuis Wix");

    // Préparer la requête pour Python
    const form = new FormData();
    form.append(
      "file",
      fs.createReadStream(req.file.path),
      req.file.originalname
    );

    // Appel à Python FastAPI
    const response = await axios.post(
      `${PYTHON_ENGINE_URL}/fingerprint`,
      form,
      { headers: form.getHeaders() }
    );

    // Supprimer le fichier temporaire
    fs.unlinkSync(req.file.path);

    // Retourner la réponse à Wix
    res.json(response.data);

  } catch (err) {
    console.error("❌ Erreur Node:", err.message);
    res.status(500).json({ status: "error", message: err.message });
  }
});

// Lancer le serveur Node
app.listen(3000, () => {
  console.log("🚀 Node API démarrée sur le port 3000");
});
