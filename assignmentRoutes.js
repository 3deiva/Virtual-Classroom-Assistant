const express = require("express");
const multer = require("multer");
const path = require("path");
const Assignment = require("./assignment"); // Adjust the path as needed
const Submission = require("./Submission"); // Adjust the path as needed
const Student = require("./Student"); // Adjust the path as needed
const router = express.Router();

// Configure multer for file uploads
const upload = multer({
  dest: "uploads/", // Directory for storing uploaded files
  limits: { fileSize: 10 * 1024 * 1024 }, // Limit file size to 10MB
});

// Create a new assignment
router.post("/assignments", async (req, res) => {
  try {
    const { assignmentName, course, staffName, deadline, password } = req.body;

    // Ensure all required fields are present
    if (!assignmentName || !course || !staffName || !deadline || !password) {
      return res.status(400).json({ message: "All fields are required." });
    }

    // Ensure deadline is a valid date
    if (isNaN(new Date(deadline))) {
      return res.status(400).json({ message: "Invalid deadline date." });
    }

    // Create new assignment
    const assignment = new Assignment({
      assignmentName,
      course,
      staffName,
      deadline: new Date(deadline),
      password,
    });

    await assignment.save();
    res.status(201).json(assignment);
  } catch (error) {
    console.error("Error creating assignment:", error);
    res.status(500).json({ message: "Error creating assignment." });
  }
});

// Get all assignments
router.get("/assignments", async (req, res) => {
  try {
    const assignments = await Assignment.find();
    res.status(200).json(assignments);
  } catch (error) {
    console.error("Error fetching assignments:", error);
    res.status(500).json({ message: "Error fetching assignments." });
  }
});

// Delete an assignment
router.delete("/assignments/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const assignment = await Assignment.findByIdAndDelete(id);

    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found." });
    }

    res.status(200).json({ message: "Assignment deleted successfully." });
  } catch (error) {
    console.error("Error deleting assignment:", error);
    res.status(500).json({ message: "Error deleting assignment." });
  }
});

// Verify assignment password
router.post("/verifyAssignmentAccess", async (req, res) => {
  const { assignmentId, password } = req.body;

  try {
    // Find the assignment by its ID
    const assignment = await Assignment.findById(assignmentId);

    if (!assignment) {
      return res
        .status(404)
        .json({ success: false, message: "Assignment not found" });
    }

    // Check if the provided password matches the assignment's password
    if (assignment.password !== password) {
      return res
        .status(400)
        .json({ success: false, message: "Incorrect password" });
    }

    // If both checks pass, return success
    res.json({
      success: true,
      message: "Password verified",
      assignmentId: assignment._id,
    });
  } catch (error) {
    console.error("Error verifying assignment access:", error);
    res
      .status(500)
      .json({ success: false, message: "Error verifying assignment access" });
  }
});

// Handle assignment submission
router.post("/assignments/submit", upload.single("file"), async (req, res) => {
  try {
    const { studentName, rollNo, assignmentName, assignmentId, driveLink } =
      req.body;
    const file = req.file;

    // Ensure all required fields are present
    if (
      !studentName ||
      !rollNo ||
      !assignmentName ||
      !assignmentId ||
      !driveLink
    ) {
      return res
        .status(400)
        .json({ success: false, message: "All fields are required." });
    }

    // Check if the student exists
    const student = await Student.findOne({ rollNo, name: studentName });
    if (!student) {
      return res.status(400).json({
        success: false,
        message: "Student not found or details do not match.",
      });
    }

    // Check if a submission already exists for this student and assignment
    const existingSubmission = await Submission.findOne({
      rollNo,
      assignmentId,
    });

    if (existingSubmission) {
      // Update existing submission
      existingSubmission.assignmentName = assignmentName;
      existingSubmission.driveLink = driveLink;
      existingSubmission.file = file ? file.path : existingSubmission.file; // Update file path if new file is uploaded
      await existingSubmission.save();
      return res
        .status(200)
        .json({ success: true, message: "Submission updated successfully." });
    } else {
      // Create a new submission record
      const submission = new Submission({
        studentName,
        rollNo,
        assignmentName,
        assignmentId,
        driveLink,
        file: file ? file.path : null, // Save file path if file is uploaded
      });

      await submission.save();
      return res
        .status(201)
        .json({ success: true, message: "Submission successful." });
    }
  } catch (error) {
    console.error("Error handling assignment submission:", error);
    res
      .status(500)
      .json({ success: false, message: "Error submitting assignment." });
  }
});

// Get pending submissions by assignment ID
router.get("/submissions/:assignmentId", async (req, res) => {
  const { assignmentId } = req.params;

  try {
    const submissions = await Submission.find({
      assignmentId,
      verified: "pending",
    });
    res.json(submissions);
  } catch (error) {
    console.error("Error fetching submissions:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Verify submission
router.post("/verifySubmission", async (req, res) => {
  const { submissionId, teacherName, rollNo } = req.body;

  try {
    const submission = await Submission.findById(submissionId);
    if (submission) {
      // Update the status to 'verified'
      submission.verified = "verified";
      submission.verifiedBy = { name: teacherName, rollNo: rollNo };
      await submission.save();
      res.json({ success: true });
    } else {
      res.status(404).json({ success: false, message: "Submission not found" });
    }
  } catch (error) {
    console.error("Error verifying submission:", error);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

// Reject submission
// Reject submission
router.post("/rejectSubmission", async (req, res) => {
  const { submissionId, teacherName, rollNo } = req.body;

  try {
    const submission = await Submission.findById(submissionId);
    if (!submission) {
      return res
        .status(404)
        .json({ success: false, message: "Submission not found" });
    }

    // Mark as rejected
    submission.verified = "rejected"; // Update field to 'rejected'
    submission.verifiedBy = { name: teacherName, rollNo: rollNo };
    await submission.save();

    res.json({ success: true, message: "Submission rejected successfully" });
  } catch (error) {
    console.error("Error rejecting submission:", error);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

// Get assignment status for a particular student
router.delete("/assignments/:id", async (req, res) => {
  try {
    await Assignment.findByIdAndDelete(req.params.id);
    res.status(200).json({ message: "Assignment deleted" });
  } catch (error) {
    res.status(500).json({ message: "Error deleting assignment" });
  }
});

router.get("/activeAssignments", async (req, res) => {
  try {
    const now = new Date();
    const assignments = await Assignment.find();

    // Mark assignments as closed if the deadline has passed
    const updatedAssignments = assignments.map((assignment) => {
      const deadline = new Date(assignment.deadline);
      return {
        ...assignment.toObject(),
        isClosed: now > deadline,
      };
    });

    res.status(200).json(updatedAssignments);
  } catch (error) {
    console.error("Error fetching assignments:", error);
    res.status(500).json({ message: "Error fetching assignments." });
  }
});

// Verify assignment access by assignment name and password

// routes/assignments.js

// Verify assignment and fetch submissions
// Verify assignment and fetch submissions
router.post("/verifyAndFetchSubmissions", async (req, res) => {
  const { assignmentName, password } = req.body;

  console.log("Received request to verify assignment");
  console.log("Assignment Name:", assignmentName);
  console.log("Password:", password);
  s;
  try {
    // Find the assignment by name and password
    const assignment = await Assignment.findOne({ assignmentName, password });

    if (!assignment) {
      console.log("No assignment found with provided name and password");
      return res.status(404).json({
        success: false,
        message: "Assignment not found or incorrect password",
      });
    }

    // Log the assignmentId to the console
    console.log("Assignment ID:", assignment._id);

    // Find submissions for the found assignment
    const submissions = await Submission.find({ assignmentId: assignment._id });

    // Prepare the response with submission details
    const result = submissions.map((submission) => ({
      studentName: submission.studentName,
      rollNo: submission.rollNo,
      verifiedStatus: submission.verified,
    }));

    res.json({ success: true, submissions: result });
  } catch (error) {
    console.error("Error fetching assignment status:", error);
    res.status(500).json({ success: false, message: "Internal server error" });
  }
});

module.exports = router;
