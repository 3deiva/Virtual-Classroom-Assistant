const express = require("express");
const multer = require("multer");
const Teacher = require("./Teacher");
const Student = require("./Student");
const router = express.Router();

// Configure multer for file upload handling
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(
      null,
      file.fieldname +
        "-" +
        uniqueSuffix +
        "." +
        file.originalname.split(".").pop()
    );
  },
});

const upload = multer({ storage: storage });

// Signup route
router.post("/signup", upload.single("faceData"), async (req, res) => {
  try {
    const { name, age, phoneNo, userId, email, role, course, rollNo } =
      req.body;
    const faceData = req.file ? req.file.filename : null;

    // Check if user already exists
    let existingUser;
    if (role === "teacher") {
      existingUser = await Teacher.findOne({
        $or: [{ userId }, { email }, { phoneNo }],
      });
    } else if (role === "student") {
      existingUser = await Student.findOne({
        $or: [{ userId }, { email }, { phoneNo }],
      });
    }

    if (existingUser) {
      let errorMessage = "User with ";
      if (existingUser.userId === userId) {
        errorMessage += "ID ";
      }
      if (existingUser.email === email) {
        errorMessage += errorMessage.includes("ID") ? ", email " : "email ";
      }
      if (existingUser.phoneNo === phoneNo) {
        errorMessage +=
          errorMessage.includes("ID") || errorMessage.includes("email")
            ? ", phone number "
            : "phone number ";
      }
      errorMessage += "already exists.";
      return res.status(400).send(errorMessage);
    }

    // Create new user based on role
    if (role === "teacher") {
      const newTeacher = new Teacher({
        name,
        age,
        phoneNo,
        userId,
        email,
        faceData,
        course,
      });
      await newTeacher.save();
    } else if (role === "student") {
      const newStudent = new Student({
        name,
        age,
        phoneNo,
        userId,
        email,
        faceData,
        course,
        rollNo,
      });
      await newStudent.save();
    }

    res.status(201).send("User created successfully");
  } catch (error) {
    console.error("Error creating user:", error);
    res.status(500).send("Error creating user");
  }
});

// Login route
router.post("/login", async (req, res) => {
  try {
    const { name, email, phoneNo, role } = req.body;

    if (role === "teacher") {
      let user = await Teacher.findOne({ name, email, phoneNo });
      if (user) {
        res.redirect(`/teacherDashboard.html?phoneNo=${phoneNo}`);
      } else {
        res.status(400).send("Teacher not found");
      }
    } else if (role === "student") {
      let user = await Student.findOne({ name, email, phoneNo });
      if (user) {
        res.redirect(`/studentDashboard.html?phoneNo=${phoneNo}`);
      } else {
        res.status(400).send("Student not found");
      }
    } else {
      res.status(400).send("Invalid role specified");
    }
  } catch (error) {
    console.error("Error during login:", error);
    res.status(500).send("Error during login");
  }
});

// Fetch teacher details
router.get("/teacher/:phoneNo", async (req, res) => {
  try {
    const phoneNo = req.params.phoneNo;
    const teacher = await Teacher.findOne({ phoneNo });

    if (!teacher) {
      return res.status(404).send("Teacher not found");
    }

    res.json(teacher);
  } catch (error) {
    console.error("Error fetching teacher details:", error);
    res.status(500).send("Error fetching teacher details");
  }
});

// Update teacher profile route
router.put(
  "/updateTeacher/:phoneNo",
  upload.single("faceData"),
  async (req, res) => {
    try {
      const phoneNo = req.params.phoneNo;
      const { name, age, email, course } = req.body;
      const faceData = req.file ? req.file.filename : null;

      const updatedTeacher = await Teacher.findOneAndUpdate(
        { phoneNo: phoneNo },
        { name, age, email, course, faceData },
        { new: true }
      );

      if (!updatedTeacher) {
        return res.status(404).send("Teacher not found");
      }

      res.status(200).send("Teacher profile updated successfully");
    } catch (error) {
      console.error("Error updating teacher profile:", error);
      res.status(500).send("Error updating teacher profile");
    }
  }
);

// Fetch student details
router.get("/student/:phoneNo", async (req, res) => {
  try {
    const phoneNo = req.params.phoneNo;
    const student = await Student.findOne({ phoneNo });

    if (!student) {
      return res.status(404).send("Student not found");
    }

    res.json(student);
  } catch (error) {
    console.error("Error fetching student details:", error);
    res.status(500).send("Error fetching student details");
  }
});

// Update student profile route
router.put(
  "/updateStudent/:phoneNo",
  upload.single("faceData"),
  async (req, res) => {
    try {
      const phoneNo = req.params.phoneNo;
      const { name, age, email, course, rollNo } = req.body;
      const faceData = req.file ? req.file.filename : null;

      const updatedStudent = await Student.findOneAndUpdate(
        { phoneNo: phoneNo },
        { name, age, email, course, rollNo, faceData },
        { new: true }
      );

      if (!updatedStudent) {
        return res.status(404).send("Student not found");
      }

      res.status(200).send("Student profile updated successfully");
    } catch (error) {
      console.error("Error updating student profile:", error);
      res.status(500).send("Error updating student profile");
    }
  }
);

// Fetch student by phone number
router.get("/api/getStudentByPhone/:phoneNo", async (req, res) => {
  try {
    const student = await Student.findOne({ phoneNo: req.params.phoneNo });
    if (student) {
      res.json(student);
    } else {
      res.status(404).json({ message: "Student not found" });
    }
  } catch (error) {
    res.status(500).json({ message: "Server error" });
  }
});

module.exports = router;
