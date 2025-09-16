const cloudinary = require("cloudinary").v2;

cloudinary.config({
  secure: true, // secure URLs (https)
});

module.exports = cloudinary;
