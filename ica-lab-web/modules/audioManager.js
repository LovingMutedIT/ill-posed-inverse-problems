/**
 * Audio Manager Module
 * Handles sound library, audio loading, processing, and playback
 */

import { Stats, AudioUtils, generateId, showNotification } from './utils.js';

export class AudioManager {
    constructor() {
        this.audioContext = null;
        this.sounds = new Map(); // Map<id, SoundObject>
        this.manifest = null;
        this.targetSampleRate = 16000;
        this.maxDuration = 20; // seconds
        this.currentlyPlaying = null;
    }

    /**
     * Initialize audio context and load manifest
     */
    async initialize() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Load manifest
            const response = await fetch('manifest.json');
            this.manifest = await response.json();
            
            this.targetSampleRate = this.manifest.settings.sampleRate;
            this.maxDuration = this.manifest.settings.maxDuration;
            
            // Load preset sounds
            await this.loadPresetSounds();
            
            return true;
        } catch (error) {
            console.error('Failed to initialize AudioManager:', error);
            showNotification('Failed to initialize audio system', 'error');
            return false;
        }
    }

    /**
     * Load all preset sounds from manifest
     */
    async loadPresetSounds() {
        const promises = this.manifest.presetSounds.map(preset => 
            this.loadPresetSound(preset)
        );
        
        await Promise.all(promises);
    }

    /**
     * Load individual preset sound
     * @param {Object} preset - Preset sound metadata
     */
    async loadPresetSound(preset) {
        try {
            // For now, generate synthetic audio since we don't have actual files
            const audioData = this.generateSyntheticAudio(preset.id);
            
            const sound = {
                id: preset.id,
                name: preset.name,
                type: 'preset',
                data: audioData,
                sampleRate: this.targetSampleRate,
                duration: audioData.length / this.targetSampleRate,
                kurtosis: Stats.kurtosis(audioData),
                thumbnailPath: preset.thumbnailPath,
                isValid: true
            };
            
            // Validate non-Gaussianity
            sound.isValid = this.validateNonGaussianity(sound.kurtosis);
            
            this.sounds.set(preset.id, sound);
        } catch (error) {
            console.error(`Failed to load preset sound ${preset.name}:`, error);
        }
    }

    /**
     * Generate synthetic audio for demonstration
     * @param {string} type - Type of synthetic signal
     * @returns {Float32Array}
     */
    generateSyntheticAudio(type) {
        const duration = 5; // 5 seconds
        const length = this.targetSampleRate * duration;
        const data = new Float32Array(length);
        
        switch (type) {
            case 'sine440':
                // Pure sine wave at 440 Hz
                for (let i = 0; i < length; i++) {
                    data[i] = Math.sin(2 * Math.PI * 440 * i / this.targetSampleRate);
                }
                break;
                
            case 'chirp':
                // Chirp signal (frequency sweep)
                const f0 = 200;
                const f1 = 2000;
                for (let i = 0; i < length; i++) {
                    const t = i / this.targetSampleRate;
                    const freq = f0 + (f1 - f0) * t / duration;
                    data[i] = Math.sin(2 * Math.PI * freq * t);
                }
                break;
                
            case 'square':
                // Square wave
                for (let i = 0; i < length; i++) {
                    const t = i / this.targetSampleRate;
                    data[i] = Math.sin(2 * Math.PI * 100 * t) > 0 ? 1 : -1;
                }
                break;
                
            case 'noise':
                // White noise (Gaussian)
                for (let i = 0; i < length; i++) {
                    // Box-Muller transform for Gaussian noise
                    const u1 = Math.random();
                    const u2 = Math.random();
                    data[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                }
                break;
                
            default:
                // Default to sine wave
                for (let i = 0; i < length; i++) {
                    data[i] = Math.sin(2 * Math.PI * 440 * i / this.targetSampleRate);
                }
        }
        
        // Normalize
        return Stats.normalize(data);
    }

    /**
     * Process uploaded audio file
     * @param {File} file - Audio file
     * @returns {Promise<Object>} Sound object
     */
    async processUploadedAudio(file) {
        try {
            // Read file as array buffer
            const arrayBuffer = await file.arrayBuffer();
            
            // Decode audio
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            
            // Convert to mono
            let monoData = AudioUtils.bufferToMono(audioBuffer);
            
            // Resample to target rate
            if (audioBuffer.sampleRate !== this.targetSampleRate) {
                monoData = AudioUtils.resample(
                    monoData,
                    audioBuffer.sampleRate,
                    this.targetSampleRate
                );
            }
            
            // Trim to max duration
            const maxSamples = this.maxDuration * this.targetSampleRate;
            if (monoData.length > maxSamples) {
                monoData = monoData.slice(0, maxSamples);
            }
            
            // Normalize
            const normalizedData = Stats.normalize(monoData);
            
            // Compute kurtosis
            const kurtosis = Stats.kurtosis(normalizedData);
            
            // Create sound object
            const sound = {
                id: generateId(),
                name: file.name,
                type: 'uploaded',
                data: normalizedData,
                sampleRate: this.targetSampleRate,
                duration: normalizedData.length / this.targetSampleRate,
                kurtosis: kurtosis,
                thumbnailPath: null,
                isValid: this.validateNonGaussianity(kurtosis)
            };
            
            this.sounds.set(sound.id, sound);
            
            return sound;
        } catch (error) {
            console.error('Failed to process uploaded audio:', error);
            throw error;
        }
    }

    /**
     * Validate if signal is sufficiently non-Gaussian
     * @param {number} kurtosis 
     * @returns {boolean}
     */
    validateNonGaussianity(kurtosis) {
        const threshold = this.manifest.settings.validationThresholds.nearlyGaussian;
        return Math.abs(kurtosis) >= threshold;
    }

    /**
     * Get validation level for kurtosis
     * @param {number} kurtosis 
     * @returns {string} 'nearly-gaussian', 'weakly', or 'strongly'
     */
    getValidationLevel(kurtosis) {
        const absKurtosis = Math.abs(kurtosis);
        const thresholds = this.manifest.settings.validationThresholds;
        
        if (absKurtosis < thresholds.nearlyGaussian) {
            return 'nearly-gaussian';
        } else if (absKurtosis < thresholds.weaklyNonGaussian) {
            return 'weakly';
        } else {
            return 'strongly';
        }
    }

    /**
     * Get sound by ID
     * @param {string} id 
     * @returns {Object}
     */
    getSound(id) {
        return this.sounds.get(id);
    }

    /**
     * Get all preset sounds
     * @returns {Array}
     */
    getPresetSounds() {
        return Array.from(this.sounds.values()).filter(s => s.type === 'preset');
    }

    /**
     * Get all uploaded sounds
     * @returns {Array}
     */
    getUploadedSounds() {
        return Array.from(this.sounds.values()).filter(s => s.type === 'uploaded');
    }

    /**
     * Remove uploaded sound
     * @param {string} id 
     */
    removeSound(id) {
        const sound = this.sounds.get(id);
        if (sound && sound.type === 'uploaded') {
            this.sounds.delete(id);
            return true;
        }
        return false;
    }

    /**
     * Play sound
     * @param {string} id 
     */
    async playSound(id) {
        const sound = this.sounds.get(id);
        if (!sound) return;
        
        // Stop currently playing sound
        if (this.currentlyPlaying) {
            this.currentlyPlaying.stop();
        }
        
        // Create buffer and play
        const buffer = AudioUtils.createBuffer(
            this.audioContext,
            sound.data,
            this.targetSampleRate
        );
        
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);
        
        source.onended = () => {
            if (this.currentlyPlaying === source) {
                this.currentlyPlaying = null;
            }
        };
        
        source.start();
        this.currentlyPlaying = source;
    }

    /**
     * Stop all playback
     */
    stopAll() {
        if (this.currentlyPlaying) {
            this.currentlyPlaying.stop();
            this.currentlyPlaying = null;
        }
    }

    /**
     * Create audio buffer from data
     * @param {Float32Array} data 
     * @returns {AudioBuffer}
     */
    createAudioBuffer(data) {
        return AudioUtils.createBuffer(this.audioContext, data, this.targetSampleRate);
    }

    /**
     * Reset audio manager (remove uploaded sounds)
     */
    reset() {
        this.stopAll();
        
        // Remove all uploaded sounds
        const uploadedIds = Array.from(this.sounds.keys()).filter(id => {
            const sound = this.sounds.get(id);
            return sound.type === 'uploaded';
        });
        
        uploadedIds.forEach(id => this.sounds.delete(id));
    }

    /**
     * Get audio context
     * @returns {AudioContext}
     */
    getAudioContext() {
        return this.audioContext;
    }
}