const pool = require('./db');

const dropAndCreateUsersTable = async () => {
  try {
    // Drop old table if exists
    await pool.query(`DROP TABLE IF EXISTS users`);
    console.log('Old users table dropped');

    // Create new table
    await pool.query(`
      CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        full_name VARCHAR(100) NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        age INT,
        gender VARCHAR(10),
        language VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    console.log('New users table created');
  } catch (err) {
    console.error('Error dropping/creating users table', err);
  }
};

module.exports = dropAndCreateUsersTable;
