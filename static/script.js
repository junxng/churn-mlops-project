// API Base URL
const API_BASE = '';

// DOM Elements
const statusInfo = document.getElementById('status-info');
const refreshStatusBtn = document.getElementById('refresh-status');
const predictionForm = document.getElementById('prediction-form');
const predictionFile = document.getElementById('prediction-file');
const predictionUpload = document.getElementById('prediction-upload');
const predictionResults = document.getElementById('prediction-results');
const predictionOutput = document.getElementById('prediction-output');
const trainingForm = document.getElementById('training-form');
const trainingFile = document.getElementById('training-file');
const trainingUpload = document.getElementById('training-upload');
const trainingResults = document.getElementById('training-results');
const trainingOutput = document.getElementById('training-output');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

// Utility Functions
function showLoading(text = 'Processing...') {
    loadingText.textContent = text;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showNotification(message, type = 'success') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
        <span>${message}</span>
    `;
    
    // Add styles if not already added
    if (!document.querySelector('.notification-styles')) {
        const style = document.createElement('style');
        style.className = 'notification-styles';
        style.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                display: flex;
                align-items: center;
                gap: 10px;
                z-index: 1001;
                transform: translateX(100%);
                transition: transform 0.3s ease;
            }
            .notification-success { border-left: 4px solid #059669; }
            .notification-error { border-left: 4px solid #dc2626; }
            .notification-info { border-left: 4px solid #2563eb; }
            .notification.show { transform: translateX(0); }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(notification);
    
    // Trigger animation
    setTimeout(() => notification.classList.add('show'), 100);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// File Upload Handlers
function setupFileUpload(uploadArea, fileInput, callback) {
    // Click to browse
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            updateUploadArea(uploadArea, file);
            if (callback) callback(file);
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            fileInput.files = files;
            updateUploadArea(uploadArea, file);
            if (callback) callback(file);
        }
    });
}

function updateUploadArea(uploadArea, file) {
    uploadArea.classList.add('has-file');
    const icon = uploadArea.querySelector('i');
    const text = uploadArea.querySelector('p');
    
    icon.className = 'fas fa-check-circle';
    text.textContent = `File selected: ${file.name}`;
}

// API Functions
async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/workflow/status`);
        const data = await response.json();
        
        let statusClass = 'ready';
        let statusText = 'Ready';
        
        if (!data.data_file_exists) {
            statusClass = 'warning';
            statusText = 'No Data';
        }
        
        statusInfo.innerHTML = `
            <div class="status-item ${data.data_file_exists ? 'success' : 'warning'}">
                <div>
                    <strong>System Status</strong>
                    <div style="font-size: 0.9rem; color: #6b7280; margin-top: 4px;">
                        ${data.message}
                    </div>
                </div>
                <span class="status-badge ${statusClass}">${statusText}</span>
            </div>
            <div class="status-item">
                <div>
                    <strong>Data File</strong>
                    <div style="font-size: 0.9rem; color: #6b7280; margin-top: 4px;">
                        Training data availability
                    </div>
                </div>
                <span class="status-badge ${data.data_file_exists ? 'ready' : 'warning'}">
                    ${data.data_file_exists ? 'Available' : 'Required'}
                </span>
            </div>
        `;
        
    } catch (error) {
        console.error('Status check failed:', error);
        statusInfo.innerHTML = `
            <div class="status-item error">
                <div>
                    <strong>Connection Error</strong>
                    <div style="font-size: 0.9rem; color: #6b7280; margin-top: 4px;">
                        Unable to connect to the API
                    </div>
                </div>
                <span class="status-badge error">Error</span>
            </div>
        `;
    }
}

async function makePrediction(formData) {
    try {
        const response = await fetch(`${API_BASE}/churn/`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.text();
            throw new Error(`Prediction failed: ${errorData}`);
        }
        
        const result = await response.json();
        
        predictionResults.style.display = 'block';
        predictionOutput.innerHTML = `
            <div style="padding: 15px; background: #ecfdf5; border: 1px solid #d1fae5; border-radius: 8px; margin-bottom: 15px;">
                <h4 style="color: #065f46; margin: 0 0 10px 0;">
                    <i class="fas fa-check-circle"></i> Prediction Completed
                </h4>
                <p style="color: #047857; margin: 0;">
                    ${result.message}
                </p>
            </div>
            <pre>${JSON.stringify(result, null, 2)}</pre>
        `;
        
        showNotification('Prediction completed successfully!', 'success');
        
    } catch (error) {
        console.error('Prediction failed:', error);
        predictionResults.style.display = 'block';
        predictionOutput.innerHTML = `
            <div style="padding: 15px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;">
                <h4 style="color: #991b1b; margin: 0 0 10px 0;">
                    <i class="fas fa-exclamation-circle"></i> Prediction Failed
                </h4>
                <p style="color: #dc2626; margin: 0;">
                    ${error.message}
                </p>
            </div>
        `;
        showNotification('Prediction failed. Please check the logs.', 'error');
    }
}

async function startTraining(formData) {
    try {
        const response = await fetch(`${API_BASE}/workflow/train`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.text();
            throw new Error(`Training failed: ${errorData}`);
        }
        
        const result = await response.json();
        
        trainingResults.style.display = 'block';
        trainingOutput.innerHTML = `
            <div style="padding: 15px; background: #ecfdf5; border: 1px solid #d1fae5; border-radius: 8px; margin-bottom: 15px;">
                <h4 style="color: #065f46; margin: 0 0 10px 0;">
                    <i class="fas fa-check-circle"></i> Training Completed
                </h4>
                <p style="color: #047857; margin: 0;">
                    ${result.message}
                </p>
                ${result.final_model_path ? `
                    <p style="color: #047857; margin: 8px 0 0 0; font-size: 0.9rem;">
                        <strong>Model Path:</strong> ${result.final_model_path}
                    </p>
                ` : ''}
            </div>
            <pre>${JSON.stringify(result, null, 2)}</pre>
        `;
        
        showNotification('Model training completed successfully!', 'success');
        
        // Refresh status after training
        setTimeout(checkStatus, 1000);
        
    } catch (error) {
        console.error('Training failed:', error);
        trainingResults.style.display = 'block';
        trainingOutput.innerHTML = `
            <div style="padding: 15px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;">
                <h4 style="color: #991b1b; margin: 0 0 10px 0;">
                    <i class="fas fa-exclamation-circle"></i> Training Failed
                </h4>
                <p style="color: #dc2626; margin: 0;">
                    ${error.message}
                </p>
            </div>
        `;
        showNotification('Training failed. Please check the logs.', 'error');
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initial status check
    checkStatus();
    
    // Setup file uploads
    setupFileUpload(predictionUpload, predictionFile, (file) => {
        document.querySelector('#prediction-form button[type="submit"]').disabled = false;
    });
    
    setupFileUpload(trainingUpload, trainingFile);
    
    // Status refresh
    refreshStatusBtn.addEventListener('click', checkStatus);
    
    // Prediction form
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!predictionFile.files[0]) {
            showNotification('Please select a file for prediction', 'error');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', predictionFile.files[0]);
        formData.append('model_version', document.getElementById('model-version').value);
        formData.append('model_name', document.getElementById('model-name').value);
        
        showLoading('Running prediction...');
        await makePrediction(formData);
        hideLoading();
    });
    
    // Training form
    trainingForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData();
        if (trainingFile.files[0]) {
            formData.append('file', trainingFile.files[0]);
        }
        
        showLoading('Starting model training...');
        await startTraining(formData);
        hideLoading();
    });
});

// Auto-refresh status every 30 seconds
setInterval(checkStatus, 30000); 