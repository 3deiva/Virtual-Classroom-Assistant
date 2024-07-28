document.addEventListener("DOMContentLoaded", () => {
  const chatBox = document.getElementById("chatBox");
  const messageInput = document.getElementById("messageInput");
  const sendButton = document.getElementById("sendButton");

  sendButton.addEventListener("click", sendMessage);

  function sendMessage() {
    const message = messageInput.value;
    if (message) {
      addMessage("user", message);
      messageInput.value = "";
      fetch("http://localhost:5005/webhooks/rest/webhook", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ sender: "user", message: message }),
      })
        .then((response) => response.json())
        .then((data) => {
          data.forEach((reply) => addMessage("bot", reply.text));
        })
        .catch((error) => console.error("Error:", error));
    }
  }

  function addMessage(sender, message) {
    const messageElement = document.createElement("div");
    messageElement.className = `message ${sender} message-enter`;
    messageElement.innerHTML = `<div class="messageContent">${message}</div>`;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
  }
});
