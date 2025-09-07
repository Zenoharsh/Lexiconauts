// index.js
const express = require("express");
require("dotenv").config();

const app = express();
const port = 3000;

// Middleware to parse JSON bodies
app.use(express.json());

// Import DB setup and routes
const createUsersTable = require("./dbsetup");
const authRoutes = require("./routes/auth"); // signup/login routes
app.use("/api/auth", authRoutes);

const protectedRoutes = require("./routes/protected");
app.use("/api", protectedRoutes);  // example: /api/dashboard


// Root route
app.get("/", (req, res) => {
  res.send("Hello from Express + PostgreSQL!");
});
// Start server
createUsersTable().then(() => {
  app.listen(port, () => {
    console.log(`🚀 Server running at http://localhost:${port}`);
  });
});


