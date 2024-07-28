const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const assignmentSchema = new Schema({
  assignmentName: {
    type: String,
    required: true,
  },
  course: {
    type: String,
    required: true,
  },
  staffName: {
    type: String,
    required: true,
  },
  deadline: {
    type: Date,
    required: true,
  },
  password: {
    type: String,
    required: true,
    unique: true, // Ensure the password is unique
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

// Create and export the Assignment model
const Assignment = mongoose.model("Assignment", assignmentSchema);
module.exports = Assignment;
