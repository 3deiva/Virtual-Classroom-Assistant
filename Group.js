const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const groupSchema = new Schema({
  groupName: { type: String, required: true },
  course: { type: String, required: true },
  groupNo: { type: String, required: true, unique: true },
  staffName: { type: String, required: true },
  password: { type: String, required: true },
});

const Group = mongoose.model("Group", groupSchema);
module.exports = Group;
