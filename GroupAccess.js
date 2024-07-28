const mongoose = require("mongoose");

const groupAccessSchema = new mongoose.Schema({
  groupId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true,
    ref: "Group",
  },
  studentId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true,
    ref: "Student",
  },
  accessGrantedAt: {
    type: Date,
    default: Date.now,
  },
});

const GroupAccess = mongoose.model("GroupAccess", groupAccessSchema);

module.exports = GroupAccess;
