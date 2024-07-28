const mongoose = require("mongoose");

const StudentGroupSchema = new mongoose.Schema({
  studentName: {
    type: String,
    required: true,
  },
  rollNo: {
    type: String,
    required: true,
  },
  password: {
    type: String,
    required: true,
  },
});

module.exports = mongoose.model("StudentGroup", StudentGroupSchema);
