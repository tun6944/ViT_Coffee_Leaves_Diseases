const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const uploadContent = document.querySelector('.upload-content');
const predictBtn = document.getElementById('predict-btn');
const resultsContainer = document.getElementById('results-container');
const predictionsDiv = document.getElementById('predictions');
const loadingDiv = document.getElementById('loading');

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
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove('hidden');
        uploadContent.classList.add('hidden');
        predictBtn.disabled = false;
        resultsContainer.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

predictBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // UI state
    predictBtn.disabled = true;
    resultsContainer.classList.add('hidden');
    loadingDiv.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        
        if (data.success) {
            displayResults(data.predictions);
        } else {
            alert('Error: ' + data.detail);
        }
    } catch (error) {
        console.error('Error during prediction:', error);
        alert('Failed to connect to the server.');
    } finally {
        loadingDiv.classList.add('hidden');
        predictBtn.disabled = false;
    }
});

function displayResults(predictions) {
    predictionsDiv.innerHTML = '';
    
    predictions.forEach((pred, index) => {
        const percentage = (pred.confidence * 100).toFixed(1);
        
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        // Use a short delay to trigger the bar animation CSS
        item.innerHTML = `
            <span class="pred-class">${pred.class.split(',')[0]}</span>
            <div class="pred-conf-bar-container">
                <div class="pred-conf-bar" id="bar-${index}" style="width: 0%"></div>
            </div>
            <span class="pred-value">${percentage}%</span>
        `;
        
        predictionsDiv.appendChild(item);
        
        // Trigger animation after DOM update
        setTimeout(() => {
            document.getElementById(`bar-${index}`).style.width = `${percentage}%`;
        }, 50 * index + 50); // Stagger animations
    });
    
    resultsContainer.classList.remove('hidden');
}
