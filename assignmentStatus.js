document
  .getElementById("studentForm")
  .addEventListener("submit", async (event) => {
    event.preventDefault();

    const studentName = document.getElementById("studentName").value;
    const rollNo = document.getElementById("rollNo").value;
    const statusContainer = document.getElementById("assignmentStatus");

    statusContainer.innerHTML = ""; // Clear previous status

    try {
      const response = await fetch(
        `/api/assignmentStatus/${rollNo}/${studentName}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch assignment status.");
      }

      const assignmentStatus = await response.json();

      if (assignmentStatus.length === 0) {
        statusContainer.innerHTML =
          "<p>No submissions found for the student.</p>";
      } else {
        const statusList = document.createElement("ul");
        assignmentStatus.forEach((status) => {
          const listItem = document.createElement("li");
          listItem.textContent = `${status.assignmentName}: ${status.status}`;
          statusList.appendChild(listItem);
        });
        statusContainer.appendChild(statusList);
      }
    } catch (error) {
      console.error("Error fetching assignment status:", error);
      statusContainer.innerHTML = "<p>Failed to fetch assignment status.</p>";
    }
  });
