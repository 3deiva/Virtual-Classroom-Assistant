const mongoose = require("mongoose");

const teacherSchema = new mongoose.Schema({
  name: String,
  age: Number,
  phoneNo: { type: String, unique: true },
  userId: { type: String, unique: true },
  email: { type: String, unique: true },
  faceData: String,
  course: String,
});

module.exports = mongoose.model("Teacher", teacherSchema);
