// sgroups.js

document.addEventListener("DOMContentLoaded", () => {
  const groupsList = document.getElementById("groups-list");

  async function loadGroups() {
    try {
      const response = await fetch("/api/groups/groups");
      const groups = await response.json();
      groupsList.innerHTML = ""; // Clear previous content

      groups.forEach((group) => {
        const groupDiv = document.createElement("div");
        groupDiv.classList.add("group-box");
        groupDiv.innerHTML = `
          <div class="group">
            <h3>${group.groupName}</h3>
            <p>Course: ${group.course}</p>
            <p>Staff Name: ${group.staffName}</p>
          </div>
        `;
        groupsList.appendChild(groupDiv);
      });
    } catch (error) {
      console.error("Error fetching groups:", error);
      alert("Error fetching groups");
    }
  }

  // Load groups on page load
  loadGroups();
});
