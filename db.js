// backend/db.js
const { Pool } = require("pg");
require("dotenv").config();

// Create a pool of connections
const pool = new Pool({
  connectionString: process.env.DATABASE_URL, // cloud PostgreSQL URL
  ssl: {
    rejectUnauthorized: false // cloud DB ke liye required
  }
});

module.exports = pool; 
