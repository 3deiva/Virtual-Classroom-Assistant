const express = require("express");
const router = express.Router();
const Student = require("./Student"); // Adjust path if necessary
const Group = require("./Group"); // Adjust path if necessary

// Endpoint to verify student and group access
router.post("/verifyStudent", async (req, res) => {
  const { studentName, rollNo, password, groupName } = req.body;

  try {
    // Check if the student exists
    const student = await Student.findOne({ name: studentName, rollNo });
    if (!student) {
      return res.json({ success: false, message: "Student does not exist." });
    }

    // Check if the group exists and verify password
    const group = await Group.findOne({ groupName, password });
    if (group) {
      return res.json({ success: true });
    } else {
      return res.json({
        success: false,
        message: "Incorrect group details or password.",
      });
    }
  } catch (error) {
    console.error("Error verifying student and group:", error);
    res.status(500).json({ error: "Error verifying student and group" });
  }
});

module.exports = router;
