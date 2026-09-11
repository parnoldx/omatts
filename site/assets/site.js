"use strict";

// Copy to Clipboard buttons with live status announcements
for (const button of document.querySelectorAll("[data-copy]")) {
  button.hidden = false;
  let timer;
  button.addEventListener("click", async () => {
    const feedback = document.getElementById(button.dataset.feedback);
    const code = document.getElementById(button.dataset.copy);
    if (!code) return;
    const text = code.textContent.trim().replace(/\s+/g, " ");

    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
      if (feedback) {
        feedback.textContent = "Command copied. Paste it into your terminal.";
      }
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      if (feedback) {
        feedback.textContent = "Select and copy the command with your browser.";
      }
    }

    clearTimeout(timer);
    timer = setTimeout(() => {
      button.textContent = "Copy";
      if (feedback) {
        feedback.textContent = "";
      }
    }, 4000);
  });
}
