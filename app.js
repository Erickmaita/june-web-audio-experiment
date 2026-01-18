import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.14.0';

// Configuration
env.allowLocalModels = false; // Force load from CDN
env.useBrowserCache = true;
// IMPORTANT: Disable multi-threading calls for broad compatibility (GitHub Pages)
// env.backends.onnx.wasm.numThreads = 1; 

const MODEL_NAME = 'Xenova/whisper-tiny';

// State
let transcriber = null;

// UI Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const statusContainer = document.getElementById('status-container');
const statusText = document.getElementById('status-text');
const outputContainer = document.getElementById('output-container');
const transcriptText = document.getElementById('transcript-text');
const copyBtn = document.getElementById('copy-btn');
const copyFeedback = document.getElementById('copy-feedback');
const dropContent = document.querySelector('.drop-content');

// --- Initialization ---

async function initTranscriber() {
    if (transcriber) return transcriber;

    updateStatus('Loading Whisper model (this happens once)...');

    // Initialize the pipeline
    transcriber = await pipeline('automatic-speech-recognition', MODEL_NAME);

    console.log("Transcriber loaded");
    return transcriber;
}

// --- Audio Processing ---

const MAX_DURATION_SEC = 120; // Hard limit: 2 minutes

async function convertAudio(file) {
    updateStatus('Processing audio locally...');

    // 1. Read file as ArrayBuffer
    const arrayBuffer = await file.arrayBuffer();

    // 2. Decode using native AudioContext
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();

    try {
        const decodedBuffer = await audioContext.decodeAudioData(arrayBuffer);

        // Check Duration Limit
        if (decodedBuffer.duration > MAX_DURATION_SEC) {
            throw new Error(`Audio too long (${Math.round(decodedBuffer.duration)}s). Limit is ${MAX_DURATION_SEC}s.`);
        }

        // 3. Resample to 16kHz mono using OfflineAudioContext
        const targetSampleRate = 16000;
        const targetLength = Math.ceil(decodedBuffer.duration * targetSampleRate);

        const offlineContext = new OfflineAudioContext(1, targetLength, targetSampleRate);
        const source = offlineContext.createBufferSource();
        source.buffer = decodedBuffer;
        source.connect(offlineContext.destination);
        source.start(0);

        const renderedBuffer = await offlineContext.startRendering();
        return renderedBuffer.getChannelData(0);

    } finally {
        if (audioContext.state !== 'closed') {
            await audioContext.close();
        }
    }
}

// --- Main Workflow ---

async function handleFile(file) {
    if (!file) return;

    // Reset UI
    outputContainer.classList.add('hidden');
    dropContent.classList.add('hidden');
    statusContainer.classList.remove('hidden');
    transcriptText.value = '';

    // Clear any previous error styling
    statusText.style.color = 'var(--text-primary)';

    try {
        // 1. Convert Audio
        const audioData = await convertAudio(file);

        // 2. Load Model (if needed)
        if (!transcriber) {
            updateStatus('Loading model...');
            await initTranscriber();
        }

        // 3. Transcribe (Single Pass)
        updateStatus('Transcribing...');

        const result = await transcriber(audioData, {
            language: 'english',
            chunk_length_s: 30, // Internal stride, not manual chunking loop
            stride_length_s: 5,
        });

        const fullText = result.text.trim();

        // 4. Done
        transcriptText.value = fullText;

        statusContainer.classList.add('hidden');
        dropContent.classList.remove('hidden');
        outputContainer.classList.remove('hidden');

        // Auto-copy
        navigator.clipboard.writeText(fullText).then(() => {
            showCopyFeedback();
        }).catch(() => { });

    } catch (error) {
        console.error(error);

        // Show clear error message
        statusText.textContent = error.message || 'An error occurred';
        statusText.style.color = '#ef4444'; // Red error color

        // Reset after delay
        setTimeout(() => {
            statusContainer.classList.add('hidden');
            dropContent.classList.remove('hidden');
            statusText.style.color = 'var(--text-primary)';
        }, 4000);
    }
}

function updateStatus(msg) {
    statusText.textContent = msg;
}

function showCopyFeedback() {
    copyFeedback.classList.add('show');
    setTimeout(() => copyFeedback.classList.remove('show'), 2000);
}

// --- Event Listeners ---

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

// Click Browse
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

// Copy Button
copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(transcriptText.value).then(showCopyFeedback);
});
