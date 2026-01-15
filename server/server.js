import express from "express";
import multer from "multer";
import fs from "fs";
import axios from "axios";
import FormData from "form-data";

const app = express();
const upload = multer({ dest: "/tmp" });

const PYTHON_ENGINE_URL = process.env.PYTHON_ENGINE_URL || "http://engine:8000";

app.post("/melody/fingerprint", upload.single("file"), async (req, res) => {
  try {
    console.log("📥 Audio reçu depuis Wix");

    const form = new FormData();
    form.append(
      "file",
      fs.createReadStream(req.file.path),
      req.file.originalname
    );

    const response = await axios.post(
      `${PYTHON_ENGINE_URL}/fingerprint`,
      form,
      { headers: form.getHeaders() }
    );

    fs.unlinkSync(req.file.path);

    res.json(response.data);

  } catch (err) {
    console.error("❌ Erreur Node:", err.message);
    res.status(500).json({ status: "error", message: err.message });
  }
});

app.listen(3000, () => {
  console.log("🚀 Node API démarrée sur le port 3000");
});
