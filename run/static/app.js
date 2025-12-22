const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

async function upload() {
  const fileInput = document.getElementById("fileInput");
  if (!fileInput.files.length) return;

  const file = fileInput.files[0];

  // Draw the preview image to canvas (if not already drawn)
  const preview = document.getElementById("preview");
  if (preview.src) {
    const img = new Image();
    img.onload = async () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Send to backend for prediction
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      drawBoxes(data.detections);
    };
    img.src = preview.src;
  }
}

function drawBoxes(detections) {
  ctx.lineWidth = 2;
  ctx.font = "16px Arial";
  ctx.strokeStyle = "red";
  ctx.fillStyle = "red";

  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = "";

  if (detections && detections.length > 0) {
    detections.forEach((det, idx) => {
      const [x1, y1, x2, y2] = det.bbox;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${det.class_name} (${(det.confidence * 100).toFixed(1)}%)`;
      ctx.fillText(label, x1, Math.max(y1 - 6, 15));

      resultDiv.innerHTML += `
        <p><b>ROI ${idx + 1}:</b> ${det.class_name}
        — ${(det.confidence * 100).toFixed(1)}%</p>
      `;
    });
  } else {
    resultDiv.innerHTML = "<p><i>No disease detected</i></p>";
  }
}

const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
  };
  reader.readAsDataURL(file);
});
