const jwt = require("jsonwebtoken");
const pool = require("../db"); //  PostgreSQL connection file

const authMiddleware = async (req, res, next) => {
  const authHeader = req.headers["authorization"];
  if (!authHeader) return res.status(401).json({ message: "No token provided" });

  const token = authHeader.split(" ")[1]; // Bearer <token>
  if (!token) return res.status(401).json({ message: "Invalid token" });

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);

    // PostgreSQL se poora user fetch 
    const result = await pool.query(
      "SELECT id, name, email, phone FROM users WHERE id = $1",
      [decoded.id]
    );

    if (result.rows.length === 0) return res.status(401).json({ message: "User not found" });

    req.user = result.rows[0]; // poora user object attach
    next();
  } catch (err) {
    console.error(err);
    return res.status(403).json({ message: "Token is not valid" });
  }
};

module.exports = authMiddleware;

