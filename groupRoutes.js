const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const Message = require("./Message");
const Group = require("./Group");
const Student = require("./Student");
const Teacher = require("./Teacher");

// Multer storage configuration
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    const ext = path.extname(file.originalname);
    cb(null, file.fieldname + "-" + uniqueSuffix + ext);
  },
});
const upload = multer({ storage: storage });

// Create a Group
router.post("/createGroup", async (req, res) => {
  try {
    const { groupName, course, staffName, password, groupNo } = req.body;
    // Check if group already exists
    const existingGroup = await Group.findOne({ groupNo });
    if (existingGroup) {
      return res.status(400).send(`Group number '${groupNo}' already exists.`);
    }
    // Create new group
    const newGroup = new Group({
      groupName,
      course,
      staffName,
      password,
      groupNo,
    });
    await newGroup.save();
    res.status(201).json({
      message: "Group created successfully",
      groupId: newGroup._id,
    });
  } catch (error) {
    console.error("Error creating group:", error);
    res.status(500).send("Error creating group");
  }
});

// Get all groups
router.get("/groups", async (req, res) => {
  try {
    const groups = await Group.find();
    res.json(groups);
  } catch (error) {
    console.error("Error fetching groups:", error);
    res.status(500).json({ error: "Error fetching groups" });
  }
});

// Get group by groupId
router.get("/:groupId", async (req, res) => {
  const { groupId } = req.params;
  try {
    const group = await Group.findById(groupId);
    if (!group) {
      return res
        .status(404)
        .json({ error: `Group with ID '${groupId}' not found.` });
    }
    res.json(group);
  } catch (error) {
    console.error(`Error fetching group ${groupId}:`, error);
    res.status(500).json({ error: `Error fetching group ${groupId}` });
  }
});

// Delete group by groupId
router.delete("/deleteGroup/:groupId", async (req, res) => {
  const { groupId } = req.params;
  try {
    console.log(`Attempting to delete group with ID: ${groupId}`);
    const deletedGroup = await Group.findByIdAndDelete(groupId);
    if (!deletedGroup) {
      return res
        .status(404)
        .json({ error: `Group with ID '${groupId}' not found.` });
    }
    console.log(`Deleted group: ${deletedGroup.groupName}`);
    // Delete all messages associated with the deleted group
    await Message.deleteMany({ group: groupId });
    res.json({
      message: `Group '${deletedGroup.groupName}' deleted successfully`,
    });
  } catch (error) {
    console.error(`Error deleting group ${groupId}:`, error);
    res.status(500).json({ error: `Error deleting group ${groupId}` });
  }
});

// Fetch messages for a group
router.get("/messages/:groupId", async (req, res) => {
  try {
    const messages = await Message.find({ group: req.params.groupId }).sort(
      { createdAt: 1 } // Sort messages by ascending createdAt
    );
    res.json(messages);
  } catch (error) {
    console.error("Error fetching messages:", error);
    res.status(500).json({ error: "Error fetching messages" });
  }
});

// Send a message in a group
router.post(
  "/sendMessage/:groupId",
  upload.single("file"),
  async (req, res) => {
    const { sender, text } = req.body;
    const { file } = req;

    try {
      const group = await Group.findById(req.params.groupId);
      if (!group) {
        return res
          .status(404)
          .json({ error: `Group with ID '${req.params.groupId}' not found.` });
      }

      const fileType = file
        ? file.mimetype.startsWith("image")
          ? "photo"
          : file.mimetype.startsWith("application/pdf")
          ? "pdf"
          : "text" // Default to "text" if it's an unknown file type
        : "text";

      const newMessage = new Message({
        group: req.params.groupId,
        content: text,
        fileUrl: file ? `/uploads/${file.filename}` : null,
        fileType: fileType,
        sender,
      });

      await newMessage.save();
      res.status(201).json({ message: "Message sent successfully" });
    } catch (error) {
      console.error("Error sending message:", error);
      res.status(500).json({ error: "Error sending message" });
    }
  }
);

// Delete a message
router.delete("/deleteMessage/:groupId/:messageId", async (req, res) => {
  const { groupId, messageId } = req.params;
  try {
    // Find and delete the message by messageId
    const result = await Message.findByIdAndDelete(messageId);

    if (!result) {
      return res.status(404).json({ message: "Message not found" });
    }

    // Optionally, delete the associated file from the filesystem
    if (result.fileUrl) {
      const filePath = path.join(__dirname, "..", result.fileUrl);
      fs.unlink(filePath, (err) => {
        if (err) {
          console.error("Error deleting file:", err);
        }
      });
    }

    res.status(200).json({ message: "Message deleted successfully" });
  } catch (error) {
    console.error("Error deleting message:", error);
    res.status(500).json({ message: "Error deleting message" });
  }
});

// Verify student details and group password
router.post("/verifyStudent", async (req, res) => {
  const { studentName, rollNo, password, groupName } = req.body;

  try {
    // Check if the student exists
    const student = await Student.findOne({
      name: studentName,
      rollNo: rollNo,
    });
    if (!student) {
      return res
        .status(400)
        .json({ success: false, message: "Student not found" });
    }

    // Check if the group exists and the password is correct
    const group = await Group.findOne({ groupName: groupName });
    if (!group) {
      return res
        .status(400)
        .json({ success: false, message: "Group not found" });
    }

    if (group.password !== password) {
      return res
        .status(400)
        .json({ success: false, message: "Incorrect group password" });
    }

    // If both checks pass
    res.json({ success: true, message: "Verification successful" });
  } catch (error) {
    console.error("Error verifying student:", error);
    res
      .status(500)
      .json({ success: false, message: "Error verifying student" });
  }
});

// Verify teacher details and group password

router.post("/verifyTeacher", async (req, res) => {
  const { teacherName, userId, password, groupName } = req.body;

  try {
    const teacher = await Teacher.findOne({
      name: teacherName,
      userId: userId,
    });
    if (!teacher) {
      return res
        .status(400)
        .json({ success: false, message: "Teacher not found" });
    }

    const group = await Group.findOne({
      groupName: groupName,
      password: password,
    });
    if (!group) {
      return res.status(400).json({
        success: false,
        message: "Group not found or incorrect password",
      });
    }

    res
      .status(200)
      .json({ success: true, message: "Teacher and group verified" });
  } catch (error) {
    console.error("Error verifying teacher and group:", error);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

module.exports = router;
