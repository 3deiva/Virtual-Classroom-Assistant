const mongoose = require("mongoose");

const messageSchema = new mongoose.Schema({
  content: {
    type: String, // For text messages
  },
  fileUrl: {
    type: String, // For storing file URLs (e.g., for PDFs or photos)
  },
  fileType: {
    type: String, // To distinguish between text, PDF, photo, etc.
    enum: ["text", "pdf", "photo", "document"],
    default: "text",
  },
  sender: {
    type: String, // Store sender's name directly
    required: true,
  },
  group: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Group", // Reference to the Group model
    required: true,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

const Message = mongoose.model("Message", messageSchema);

module.exports = Message;
