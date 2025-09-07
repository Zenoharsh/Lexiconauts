require("dotenv").config();
const { Pool } = require("pg");

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

(async () => {
  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(50),
        email VARCHAR(50)
      )
    `);
    console.log("✅ Table created successfully");
    process.exit();
  } catch (err) {
    console.error("❌ Error creating table:", err);
    process.exit(1);
  }
})();
