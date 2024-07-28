const express = require("express");
const mongoose = require("mongoose");
const path = require("path");
const authRoutes = require("./auth");
const groupRoutes = require("./groupRoutes");
const studentRoutes = require("./studentRoutes");
const assignmentRoutes = require("./assignmentRoutes");

const app = express();
const port = process.env.PORT || 5000;

// Connect to MongoDB
mongoose
  .connect(process.env.MONGO_URI || "mongodb://localhost:27017/vir", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => {
    console.log("Connected to MongoDB");
  })
  .catch((error) => {
    console.error("Error connecting to MongoDB:", error);
  });

// Middleware for handling JSON, URL-encoded bodies
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));

// Serve static files from the root directory
app.use(express.static(path.join(__dirname)));

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/groups", groupRoutes);
app.use("/api/students", studentRoutes);
app.use("/api", assignmentRoutes);

// Error handling middleware (optional)
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send("Something went wrong!");
});

// Start server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
