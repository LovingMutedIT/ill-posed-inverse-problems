/**
 * Main Application Controller
 * Coordinates all modules and handles UI interactions
 */

import { AudioManager } from './modules/audioManager.js';
import { Validator } from './modules/validator.js';
import { Mixer } from './modules/mixer.js';
import { ICAController } from './modules/icaController.js';
import { SourceMatcher } from './modules/matcher.js';
import { AudioUtils, showNotification } from './modules/utils.js';

class ICALaboratory {
    constructor() {
        this.audioManager = null;
        this.validator = null;
        this.mixer = null;
        this.icaController = null;
        this.matcher = null;
        
        this.selectedSounds = new Set();
        this.mixedSignals = null;
        this.separatedSources = null;
        this.originalSoundObjects = [];
        
        this.state = {
            initialized: false,
            mixed: false,
            separated: false
        };
    }

    /**
     * Initialize application
     */
    async initialize() {
        try {
            showNotification('Initializing ICA Laboratory...', 'info', 2000);
            
            // Initialize audio manager
            this.audioManager = new AudioManager();
            const success = await this.audioManager.initialize();
            
            if (!success) {
                throw new Error('Failed to initialize audio system');
            }
            
            // Get settings from manifest
            const manifest = this.audioManager.manifest;
            
            // Initialize modules
            this.validator = new Validator(manifest.settings);
            this.mixer = new Mixer();
            this.icaController = new ICAController(manifest.settings);
            this.matcher = new SourceMatcher();
            
            // Setup UI
            this.setupEventListeners();
            this.renderSoundLibrary();
            this.updateUI();
            
            this.state.initialized = true;
            
            showNotification('System initialized successfully', 'success');
            
        } catch (error) {
            console.error('Initialization failed:', error);
            showNotification(`Initialization failed: ${error.message}`, 'error', 5000);
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // File upload
        document.getElementById('fileUpload').addEventListener('change', (e) => {
            this.handleFileUpload(e);
        });
        
        // Algorithm selection
        document.querySelectorAll('input[name="algorithm"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.handleAlgorithmChange(e.target.value);
            });
        });
        
        // Mix button
        document.getElementById('mixBtn').addEventListener('click', () => {
            this.handleMix();
        });
        
        // Separate button
        document.getElementById('separateBtn').addEventListener('click', () => {
            this.handleSeparate();
        });
        
        // Reset button
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.handleReset();
        });
    }

    /**
     * Render sound library
     */
    renderSoundLibrary() {
        const soundGrid = document.getElementById('soundGrid');
        soundGrid.innerHTML = '';
        
        const presetSounds = this.audioManager.getPresetSounds();
        
        presetSounds.forEach(sound => {
            const card = this.createSoundCard(sound);
            soundGrid.appendChild(card);
        });
    }

    /**
     * Create sound card element
     */
    createSoundCard(sound) {
        const card = document.createElement('div');
        card.className = 'sound-card';
        card.dataset.soundId = sound.id;
        
        if (!sound.isValid) {
            card.classList.add('invalid');
        }
        
        // Thumbnail
        const thumbnail = document.createElement('div');
        thumbnail.className = 'sound-thumbnail';
        thumbnail.innerHTML = '<span class="sound-thumbnail-placeholder">🎵</span>';
        
        // Info
        const info = document.createElement('div');
        info.className = 'sound-info';
        
        const name = document.createElement('div');
        name.className = 'sound-name';
        name.textContent = sound.name;
        
        const meta = document.createElement('div');
        meta.className = 'sound-meta';
        
        const duration = document.createElement('span');
        duration.className = 'sound-duration';
        duration.textContent = AudioUtils.formatDuration(sound.duration);
        
        const kurtosis = document.createElement('span');
        kurtosis.className = 'sound-kurtosis';
        
        const kurtValue = document.createElement('span');
        kurtValue.className = 'kurtosis-value';
        kurtValue.textContent = `κ=${Math.abs(sound.kurtosis).toFixed(2)}`;
        
        const indicator = document.createElement('span');
        indicator.className = 'gaussian-indicator';
        const level = this.audioManager.getValidationLevel(sound.kurtosis);
        if (level === 'nearly-gaussian') {
            indicator.classList.add('nearly-gaussian');
        } else if (level === 'weakly') {
            indicator.classList.add('weakly');
        }
        
        kurtosis.appendChild(kurtValue);
        kurtosis.appendChild(indicator);
        
        meta.appendChild(duration);
        meta.appendChild(kurtosis);
        
        info.appendChild(name);
        info.appendChild(meta);
        
        // Controls
        const controls = document.createElement('div');
        controls.className = 'sound-controls';
        
        const playBtn = document.createElement('button');
        playBtn.className = 'sound-play-btn';
        playBtn.textContent = '▶ Play';
        playBtn.onclick = (e) => {
            e.stopPropagation();
            this.audioManager.playSound(sound.id);
        };
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'sound-select-checkbox';
        checkbox.disabled = !sound.isValid;
        checkbox.onchange = (e) => {
            e.stopPropagation();
            this.handleSoundSelection(sound.id, e.target.checked);
        };
        
        controls.appendChild(playBtn);
        controls.appendChild(checkbox);
        
        card.appendChild(thumbnail);
        card.appendChild(info);
        card.appendChild(controls);
        
        // Click to toggle selection
        card.onclick = () => {
            if (sound.isValid) {
                checkbox.checked = !checkbox.checked;
                this.handleSoundSelection(sound.id, checkbox.checked);
            }
        };
        
        return card;
    }

    /**
     * Handle file upload
     */
    async handleFileUpload(event) {
        const files = Array.from(event.target.files);
        
        for (const file of files) {
            try {
                showNotification(`Processing ${file.name}...`, 'info', 2000);
                
                const sound = await this.audioManager.processUploadedAudio(file);
                
                this.renderUploadedSound(sound);
                
                if (!sound.isValid) {
                    showNotification(
                        `${sound.name} is nearly Gaussian (κ=${Math.abs(sound.kurtosis).toFixed(2)}) and cannot be used`,
                        'warning',
                        4000
                    );
                }
                
            } catch (error) {
                showNotification(`Failed to process ${file.name}`, 'error');
            }
        }
        
        // Reset file input
        event.target.value = '';
    }

    /**
     * Render uploaded sound
     */
    renderUploadedSound(sound) {
        const uploadedGrid = document.getElementById('uploadedSoundsGrid');
        const card = this.createSoundCard(sound);
        uploadedGrid.appendChild(card);
    }

    /**
     * Handle sound selection
     */
    handleSoundSelection(soundId, selected) {
        if (selected) {
            this.selectedSounds.add(soundId);
        } else {
            this.selectedSounds.delete(soundId);
        }
        
        // Update card visuals
        const card = document.querySelector(`[data-sound-id="${soundId}"]`);
        if (card) {
            if (selected) {
                card.classList.add('selected');
            } else {
                card.classList.remove('selected');
            }
        }
        
        this.updateUI();
    }

    /**
     * Handle algorithm change
     */
    handleAlgorithmChange(algorithm) {
        this.icaController.setAlgorithm(algorithm);
        
        // Show warning for JADE with many sources
        const warning = document.getElementById('algorithmWarning');
        if (algorithm === 'jade' && this.selectedSounds.size > 6) {
            warning.style.display = 'flex';
        } else {
            warning.style.display = 'none';
        }
    }

    /**
     * Handle mix
     */
    async handleMix() {
        try {
            // Get selected sound objects
            const selectedSoundObjects = Array.from(this.selectedSounds).map(id => 
                this.audioManager.getSound(id)
            );
            
            // Validate
            const validation = this.validator.validate(selectedSoundObjects);
            
            if (!validation.isValid) {
                showNotification(validation.errors.join('; '), 'error', 5000);
                return;
            }
            
            if (validation.warnings.length > 0) {
                showNotification(validation.warnings.join('; '), 'warning', 4000);
            }
            
            // Mix signals
            showNotification('Mixing signals...', 'info', 2000);
            
            this.originalSoundObjects = selectedSoundObjects;
            const mixResult = this.mixer.mix(selectedSoundObjects);
            this.mixedSignals = mixResult.mixedSignals;
            
            // Display mixed signals
            this.displayMixedSignals(mixResult);
            
            this.state.mixed = true;
            this.updateUI();
            
            showNotification('Signals mixed successfully', 'success');
            
        } catch (error) {
            console.error('Mixing failed:', error);
            showNotification(`Mixing failed: ${error.message}`, 'error');
        }
    }

    /**
     * Display mixed signals
     */
    displayMixedSignals(mixResult) {
        const container = document.getElementById('mixedSignalsContainer');
        const status = document.getElementById('mixingStatus');
        
        status.style.display = 'none';
        container.style.display = 'block';
        container.innerHTML = '';
        
        mixResult.mixedSignals.forEach((signal, index) => {
            const item = this.createSignalItem(`Mixture ${index + 1}`, signal, 'mixed');
            container.appendChild(item);
        });
    }

    /**
     * Handle separate
     */
    async handleSeparate() {
        try {
            const progressContainer = document.getElementById('separationProgress');
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            const status = document.getElementById('separationStatus');
            
            status.style.display = 'none';
            progressContainer.style.display = 'block';
            
            // Progress callback
            const updateProgress = (progress) => {
                progressFill.style.width = `${progress.progress * 100}%`;
                progressText.textContent = progress.message || 'Processing...';
            };
            
            // Perform ICA separation
            const result = await this.icaController.separate(
                this.mixedSignals.map(s => new Float32Array(s)),
                updateProgress
            );
            
            // Match sources
            const originalSources = this.mixer.getOriginalSources();
            const originalNames = this.originalSoundObjects.map(s => s.name);
            
            const matchResults = this.matcher.match(
                originalSources,
                result.sources,
                originalNames
            );
            
            // Apply sign correction and reorder
            let corrected = this.matcher.applySignCorrection(result.sources);
            corrected = this.matcher.reorderSources(corrected);
            
            this.separatedSources = corrected;
            
            // Display results
            progressContainer.style.display = 'none';
            this.displaySeparatedSources(corrected, matchResults);
            
            this.state.separated = true;
            this.updateUI();
            
            const avgQuality = this.matcher.getAverageSeparationQuality();
            showNotification(
                `Separation complete. Average quality: ${(avgQuality * 100).toFixed(1)}%`,
                'success'
            );
            
        } catch (error) {
            console.error('Separation failed:', error);
            showNotification(`Separation failed: ${error.message}`, 'error');
            
            document.getElementById('separationProgress').style.display = 'none';
            document.getElementById('separationStatus').style.display = 'block';
        }
    }

    /**
     * Display separated sources
     */
    displaySeparatedSources(sources, matchResults) {
        const container = document.getElementById('separatedSignalsContainer');
        container.style.display = 'block';
        container.innerHTML = '';
        
        sources.forEach((source, index) => {
            const match = matchResults.find(m => m.matchedIndex === index);
            const label = match 
                ? `${match.matchedName} (${match.confidence.toFixed(1)}% match)`
                : `Source ${index + 1}`;
            
            const item = this.createSignalItem(label, source, 'separated', match);
            container.appendChild(item);
        });
    }

    /**
     * Create signal item element
     */
    createSignalItem(title, signal, type, matchInfo = null) {
        const item = document.createElement('div');
        item.className = 'signal-item';
        
        const header = document.createElement('div');
        header.className = 'signal-header';
        
        const titleElem = document.createElement('div');
        titleElem.className = 'signal-title';
        titleElem.textContent = title;
        
        const label = document.createElement('div');
        label.className = 'signal-label';
        label.textContent = type.toUpperCase();
        
        if (matchInfo) {
            const quality = document.createElement('span');
            quality.style.marginLeft = '8px';
            quality.textContent = `[${matchInfo.quality}]`;
            label.appendChild(quality);
        }
        
        header.appendChild(titleElem);
        header.appendChild(label);
        
        // Waveform visualization placeholder
        const waveform = document.createElement('div');
        waveform.className = 'signal-waveform';
        this.drawMiniWaveform(waveform, signal);
        
        // Audio controls
        const controls = document.createElement('div');
        controls.className = 'signal-controls';
        
        const audio = document.createElement('audio');
        audio.controls = true;
        
        // Create audio blob
        const audioBuffer = this.audioManager.createAudioBuffer(signal);
        const blob = this.audioBufferToBlob(audioBuffer);
        audio.src = URL.createObjectURL(blob);
        
        controls.appendChild(audio);
        
        item.appendChild(header);
        item.appendChild(waveform);
        item.appendChild(controls);
        
        return item;
    }

    /**
     * Draw mini waveform visualization
     */
    drawMiniWaveform(container, signal) {
        const canvas = document.createElement('canvas');
        canvas.width = container.clientWidth || 600;
        canvas.height = 80;
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        
        const ctx = canvas.getContext('2d');
        
        // Background
        ctx.fillStyle = getComputedStyle(document.documentElement)
            .getPropertyValue('--color-bg-secondary').trim();
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Waveform
        const step = Math.max(1, Math.floor(signal.length / canvas.width));
        const amp = canvas.height / 2;
        
        ctx.strokeStyle = getComputedStyle(document.documentElement)
            .getPropertyValue('--color-accent-primary').trim();
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        
        for (let i = 0; i < canvas.width; i++) {
            const index = i * step;
            if (index >= signal.length) break;
            
            const value = signal[index];
            const x = i;
            const y = amp - value * amp * 0.8;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        container.appendChild(canvas);
    }

    /**
     * Convert audio buffer to blob
     */
    audioBufferToBlob(audioBuffer) {
        const wav = this.audioBufferToWav(audioBuffer);
        return new Blob([wav], { type: 'audio/wav' });
    }

    /**
     * Convert audio buffer to WAV format
     */
    audioBufferToWav(audioBuffer) {
        const numChannels = 1;
        const sampleRate = audioBuffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;
        
        const channelData = audioBuffer.getChannelData(0);
        const samples = new Int16Array(channelData.length);
        
        for (let i = 0; i < channelData.length; i++) {
            const s = Math.max(-1, Math.min(1, channelData[i]));
            samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numChannels * bitDepth / 8, true);
        view.setUint16(32, numChannels * bitDepth / 8, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);
        
        for (let i = 0; i < samples.length; i++) {
            view.setInt16(44 + i * 2, samples[i], true);
        }
        
        return buffer;
    }

    /**
     * Update UI state
     */
    updateUI() {
        // Selected count
        document.getElementById('selectedCount').textContent = this.selectedSounds.size;
        
        // Mix button
        const mixBtn = document.getElementById('mixBtn');
        mixBtn.disabled = this.selectedSounds.size < 2;
        
        // Separate button
        const separateBtn = document.getElementById('separateBtn');
        separateBtn.disabled = !this.state.mixed;
        
        // Reset button
        const resetBtn = document.getElementById('resetBtn');
        resetBtn.disabled = !this.state.mixed && !this.state.separated;
        
        // Algorithm warning
        const algorithm = document.querySelector('input[name="algorithm"]:checked').value;
        const warning = document.getElementById('algorithmWarning');
        if (algorithm === 'jade' && this.selectedSounds.size > 6) {
            warning.style.display = 'flex';
        } else {
            warning.style.display = 'none';
        }
    }

    /**
     * Handle reset
     */
    handleReset() {
        if (!confirm('Reset the entire system? This will clear all mixed and separated signals.')) {
            return;
        }
        
        // Stop all audio
        this.audioManager.stopAll();
        
        // Reset modules
        this.mixer.reset();
        this.icaController.reset();
        this.matcher.reset();
        
        // Clear selections
        this.selectedSounds.clear();
        document.querySelectorAll('.sound-select-checkbox').forEach(cb => {
            cb.checked = false;
        });
        document.querySelectorAll('.sound-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        // Remove uploaded sounds
        this.audioManager.reset();
        document.getElementById('uploadedSoundsGrid').innerHTML = '';
        
        // Clear displays
        document.getElementById('mixedSignalsContainer').style.display = 'none';
        document.getElementById('mixedSignalsContainer').innerHTML = '';
        document.getElementById('mixingStatus').style.display = 'block';
        
        document.getElementById('separatedSignalsContainer').style.display = 'none';
        document.getElementById('separatedSignalsContainer').innerHTML = '';
        document.getElementById('separationStatus').style.display = 'block';
        document.getElementById('separationProgress').style.display = 'none';
        
        // Reset algorithm to FastICA
        document.querySelector('input[name="algorithm"][value="fastica"]').checked = true;
        this.icaController.setAlgorithm('fastica');
        
        // Reset state
        this.state.mixed = false;
        this.state.separated = false;
        this.mixedSignals = null;
        this.separatedSources = null;
        this.originalSoundObjects = [];
        
        this.updateUI();
        
        showNotification('System reset successfully', 'success');
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new ICALaboratory();
    app.initialize();
});