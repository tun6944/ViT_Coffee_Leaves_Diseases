const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const uploadContent = document.querySelector('.upload-content');
const predictBtn = document.getElementById('predict-btn');
const resultsContainer = document.getElementById('results-container');
const resultDiv = document.getElementById('result');
const loadingDiv = document.getElementById('loading');
const canvasContainer = document.getElementById('canvas-container');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let currentFile = null;

// Handle drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', handleDrop, false);
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    handleFile(file);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    handleFile(file);
}

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    currentFile = file;
    
    // Hide previous results
    resultsContainer.classList.add('hidden');
    canvasContainer.classList.add('hidden');
    
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        dropZone.classList.remove('hidden');
        uploadContent.classList.add('hidden');
        predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

predictBtn.addEventListener('click', async () => {
    if (!currentFile || !preview.src) return;

    // UI state
    predictBtn.disabled = true;
    resultsContainer.classList.add('hidden');
    canvasContainer.classList.add('hidden');
    loadingDiv.classList.remove('hidden');
    
    // Hide the upload zone temporarily while processing
    dropZone.classList.add('hidden');

    try {
        const img = new Image();
        img.onload = async () => {
            // Setup canvas
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            // Send to backend for prediction
            const formData = new FormData();
            formData.append('file', currentFile);

            // Fetch
            const res = await fetch("http://localhost:8000/predict", {
                method: "POST",
                body: formData,
            });

            const data = await res.json();
            
            loadingDiv.classList.add('hidden');
            canvasContainer.classList.remove('hidden');
            drawBoxes(data.detections);
            predictBtn.disabled = false;
            
            dropZone.classList.remove('hidden');
            preview.classList.add('hidden');
            uploadContent.classList.remove('hidden');
            currentFile = null;
        };
        img.src = preview.src;
        
    } catch (error) {
        console.error('Error during prediction:', error);
        alert('Failed to connect to the server or process the image.');
        loadingDiv.classList.add('hidden');
        dropZone.classList.remove('hidden');
        predictBtn.disabled = false;
    }
});

function drawBoxes(detections) {
    ctx.lineWidth = 4;
    ctx.font = "bold 24px Inter, sans-serif";
    ctx.strokeStyle = "#ff4a4a"; // using a vibrant red
    ctx.fillStyle = "#ff4a4a";

    resultDiv.innerHTML = "";

    let shownCount = 0;
    if (detections && detections.length > 0) {
        detections.forEach((det, idx) => {
            if (det.confidence >= 0.4) {
                const [x1, y1, x2, y2] = det.bbox;
                
                // Draw box
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Draw label background
                const label = `${det.class_name} ${(det.confidence * 100).toFixed(1)}%`;
                const textWidth = ctx.measureText(label).width;
                ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
                ctx.fillRect(x1, Math.max(y1 - 32, 0), textWidth + 10, 32);
                
                // Draw label text
                ctx.fillStyle = "#ff4a4a";
                ctx.fillText(label, x1 + 5, Math.max(y1 - 8, 24));

                const percentage = (det.confidence * 100).toFixed(1);
                
                const item = document.createElement('div');
                item.className = 'detection-item';
                
                item.innerHTML = `
                    <span class="det-class">
                        <span class="det-roi">ROI ${idx + 1}</span>
                        ${det.class_name}
                    </span>
                    <div class="det-conf-bar-container">
                        <div class="det-conf-bar" id="bar-${idx}" style="width: 0%"></div>
                    </div>
                    <span class="det-value">${percentage}%</span>
                `;
                
                resultDiv.appendChild(item);
                
                // Trigger animation after DOM update
                setTimeout(() => {
                    const bar = document.getElementById(`bar-${idx}`);
                    if(bar) bar.style.width = `${percentage}%`;
                }, 50 * shownCount + 50);

                shownCount++;
            }
        });
    }
    
    if (shownCount === 0) {
        resultDiv.innerHTML = '<div class="no-disease">Everything looks healthy! No disease detected.</div>';
    }
    
    resultsContainer.classList.remove('hidden');
}
