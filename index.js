const express = require("express");
require("dotenv").config();

const app = express();
const port = 3000;

// Middleware to parse JSON bodies
app.use(express.json());

// Import DB setup and routes
const createUsersTable = require("./dbsetup");
const authRoutes = require("./routes/auth");
const protectedRoutes = require("./routes/protected");
const videoRoutes = require("./routes/videos");

// Routes
app.use("/api/auth", authRoutes);
app.use("/api", protectedRoutes);
app.use("/api/videos", videoRoutes); // 🔥 video routes added

// Root route
app.get("/", (req, res) => {
  res.send("Hello from Express + PostgreSQL + Cloudinary! 🚀");
});

// Start server
createUsersTable().then(() => {
  app.listen(port, () => {
    console.log(`🚀 Server running at http://localhost:${port}`);
  });
});
