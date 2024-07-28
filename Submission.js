const mongoose = require("mongoose");

const submissionSchema = new mongoose.Schema({
  studentName: { type: String, required: true },
  rollNo: { type: String, required: true },
  assignmentName: { type: String, required: true },
  assignmentId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Assignment",
    required: true,
  },
  driveLink: { type: String, required: true },
  verified: {
    type: String,
    enum: ["pending", "verified", "rejected"], // Add the possible statuses
    default: "pending", // Default status
  },
  verifiedBy: {
    name: String,
    rollNo: String,
  },
});

const Submission = mongoose.model("Submission", submissionSchema);
module.exports = Submission;
