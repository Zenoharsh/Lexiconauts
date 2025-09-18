const pool = require('./db');

const dropAndCreateUsersTable = async () => {
  try {
    // Drop old users table if exists
    await pool.query(`DROP TABLE IF EXISTS users CASCADE`);
    console.log('Old users table dropped');

    // Create new users table
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

    // Drop videos table if exists
    await pool.query(`DROP TABLE IF EXISTS videos CASCADE`);
    console.log('Old videos table dropped');

    // Create new videos table
    await pool.query(`
      CREATE TABLE videos (
        id SERIAL PRIMARY KEY,
        user_id INT REFERENCES users(id) ON DELETE CASCADE,
        original_name TEXT NOT NULL,
        url TEXT NOT NULL,
        public_id VARCHAR(255) NOT NULL,
        mimetype TEXT NOT NULL,
        size BIGINT NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    console.log('Videos table created');
  } catch (err) {
    console.error('Error dropping/creating tables', err);
  }
};

module.exports = dropAndCreateUsersTable;
