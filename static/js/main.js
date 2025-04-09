document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const clearBtn = document.getElementById('clearBtn');
    const copyBtn = document.getElementById('copyBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const output = document.getElementById('output');
    const statusLight = document.getElementById('statusLight');
    const statusText = document.getElementById('statusText');
    
    // Audio recording variables
    let mediaRecorder = null;
    let recordingStream = null;
    let audioChunks = [];
    let isRecording = false;
    let socket;
    
    // Initialize Socket.IO connection
    function initializeSocket() {
        socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to server');
            showToast('Connected to server', 'success');
            startBtn.disabled = false;
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from server');
            showToast('Disconnected from server', 'danger');
            startBtn.disabled = true;
            if (isRecording) {
                stopRecording();
            }
        });
        
        socket.on('transcription', function(data) {
            if (data.text) {
                output.textContent += (output.textContent ? '\n' : '') + data.text;
                output.scrollTop = output.scrollHeight;
            }
        });
        
        socket.on('error', function(data) {
            console.error('Server error:', data.message);
            statusText.textContent = 'Error: ' + data.message;
            showToast('Error: ' + data.message, 'danger');
            if (isRecording) {
                stopRecording();
            }
        });
    }
    
    // Start recording audio
    async function startRecording() {
        try {
            // Reset audio chunks
            audioChunks = [];
            
            console.log('Requesting microphone access...');
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });
            console.log('Microphone access granted');
            
            recordingStream = stream;
            
            // Create MediaRecorder with WAV format
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm'
            });
            
            // Handle data available event
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    console.log('Received audio chunk of size:', event.data.size);
                    audioChunks.push(event.data);
                }
            };
            
            // Handle recording stop
            mediaRecorder.onstop = async () => {
                console.log('Recording stopped, processing audio...');
                try {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    console.log('Created audio blob of size:', audioBlob.size);
                    await sendAudioToServer(audioBlob);
                } catch (error) {
                    console.error('Error processing audio:', error);
                    showToast('Error processing audio: ' + error.message, 'danger');
                    statusText.textContent = 'Error processing audio';
                }
            };
            
            // Start recording with 1-second timeslices
            mediaRecorder.start(1000);
            console.log('Started recording');
            
            isRecording = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusLight.classList.add('active');
            statusText.textContent = 'Recording...';
            showToast('Recording started', 'info');
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            statusText.textContent = 'Error: ' + error.message;
            showToast('Error accessing microphone: ' + error.message, 'danger');
            cleanupRecording();
        }
    }
    
    // Stop recording audio
    function stopRecording() {
        console.log('Stopping recording...');
        if (!isRecording) return;
        
        try {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                console.log('MediaRecorder stopped');
            }
            
            cleanupRecording();
            statusText.textContent = 'Processing...';
            showToast('Processing audio...', 'info');
        } catch (error) {
            console.error('Error stopping recording:', error);
            showToast('Error stopping recording: ' + error.message, 'danger');
            statusText.textContent = 'Error stopping recording';
            cleanupRecording();
        }
    }
    
    // Cleanup recording resources
    function cleanupRecording() {
        console.log('Cleaning up recording resources...');
        if (recordingStream) {
            recordingStream.getTracks().forEach(track => {
                track.stop();
                console.log('Audio track stopped');
            });
            recordingStream = null;
        }
        
        mediaRecorder = null;
        isRecording = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        statusLight.classList.remove('active');
        console.log('Cleanup complete');
    }
    
    // Send audio data to the server
    async function sendAudioToServer(audioBlob) {
        console.log('Sending audio to server...');
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onloadend = () => {
                try {
                    socket.emit('audio_data', { audio: reader.result });
                    console.log('Audio data sent to server');
                    statusText.textContent = 'Idle';
                    resolve();
                } catch (error) {
                    console.error('Error sending audio data:', error);
                    showToast('Error sending audio data: ' + error.message, 'danger');
                    statusText.textContent = 'Error sending audio';
                    reject(error);
                }
            };
            
            reader.onerror = (error) => {
                console.error('Error reading audio blob:', error);
                showToast('Error processing audio: ' + error.message, 'danger');
                statusText.textContent = 'Error processing audio';
                reject(error);
            };
            
            reader.readAsDataURL(audioBlob);
        });
    }
    
    // Clear the transcription output
    function clearTranscription() {
        output.textContent = '';
        showToast('Transcription cleared', 'info');
    }
    
    // Copy transcription to clipboard
    function copyToClipboard() {
        if (!output.textContent) {
            showToast('No text to copy', 'warning');
            return;
        }
        
        navigator.clipboard.writeText(output.textContent)
            .then(() => {
                showToast('Copied to clipboard', 'success');
            })
            .catch(err => {
                console.error('Failed to copy:', err);
                showToast('Failed to copy: ' + err.message, 'danger');
            });
    }
    
    // Download transcription as text file
    function downloadTranscription() {
        if (!output.textContent) {
            showToast('No text to download', 'warning');
            return;
        }
        
        const text = output.textContent;
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'transcription_' + new Date().toISOString().slice(0, 19).replace(/:/g, '-') + '.txt';
        document.body.appendChild(a);
        a.click();
        
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
        
        showToast('Downloaded transcription', 'success');
    }
    
    // Show toast notification
    function showToast(message, type) {
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }
        
        const toastId = 'toast-' + Date.now();
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-bg-${type} border-0`;
        toast.id = toastId;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', function() {
            toast.remove();
        });
    }
    
    // Event listeners
    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    clearBtn.addEventListener('click', clearTranscription);
    copyBtn.addEventListener('click', copyToClipboard);
    downloadBtn.addEventListener('click', downloadTranscription);
    
    // Initialize Socket.IO on page load
    initializeSocket();
});