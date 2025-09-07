const express = require("express");
const router = express.Router();
const authMiddleware = require("../middleware/authmiddleware");

// User profile route
router.get("/profile", authMiddleware, (req, res) => {
  res.json({
    message: "User details fetched successfully",
    user: req.user // poora user object
  });
});

module.exports = router;
