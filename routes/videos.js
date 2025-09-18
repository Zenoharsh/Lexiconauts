// routes/videos.js
const express = require("express");
const multer = require("multer");
const pool = require("../db");
const { ensureAuth } = require("../middleware/authmiddleware");
const cloudinary = require("../cloudinary");

const router = express.Router();

// Multer setup (memory storage)
const upload = multer({ storage: multer.memoryStorage() });

// ----------------- Upload Video -----------------
router.post("/upload", ensureAuth, upload.single("video"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const userId = req.user.id; // authMiddleware se set hota hai
    const description = req.body.description || null;

    // Upload to Cloudinary
    const uploadStream = cloudinary.uploader.upload_stream(
      {
        resource_type: "video",
        folder: "sports_app", // optional folder in Cloudinary
      },
      async (error, result) => {
        if (error) {
          console.error("Cloudinary upload error:", error);
          return res.status(500).json({ error: "Upload to Cloudinary failed" });
        }

        // Save in DB
        const query = `
          INSERT INTO videos (user_id, original_name, url, public_id, mimetype, size, description)
          VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id, created_at
        `;
        const values = [
          userId,
          req.file.originalname,
          result.secure_url,
          result.public_id,
          req.file.mimetype,
          req.file.size,
          description,
        ];

        const dbRes = await pool.query(query, values);
        const saved = dbRes.rows[0];

        res.status(201).json({
          message: "Video uploaded successfully",
          video: {
            id: saved.id,
            user_id: userId,
            original_name: req.file.originalname,
            url: result.secure_url,
            public_id: result.public_id,
            mimetype: req.file.mimetype,
            size: req.file.size,
            description,
            created_at: saved.created_at,
          },
        });
      }
    );

    // Pipe buffer to Cloudinary
    uploadStream.end(req.file.buffer);
  } catch (err) {
    console.error("Upload error:", err);
    res.status(500).json({ error: "Server error while uploading video" });
  }
});

// ----------------- Delete Video -----------------
router.delete("/:id", ensureAuth, async (req, res) => {
  try {
    const videoId = req.params.id;
    const userId = req.user.id;

    // Check video ownership
    const check = await pool.query(
      "SELECT * FROM videos WHERE id=$1 AND user_id=$2",
      [videoId, userId]
    );

    if (check.rows.length === 0) {
      return res.status(404).json({ error: "Video not found or not owned by user" });
    }

    const video = check.rows[0];

    // Delete from Cloudinary
    await cloudinary.uploader.destroy(video.public_id, { resource_type: "video" });

    // Delete from DB
    await pool.query("DELETE FROM videos WHERE id=$1 AND user_id=$2", [videoId, userId]);

    res.json({ message: "Video deleted successfully" });
  } catch (err) {
    console.error("Delete error:", err);
    res.status(500).json({ error: "Server error while deleting video" });
  }
});

module.exports = router;
