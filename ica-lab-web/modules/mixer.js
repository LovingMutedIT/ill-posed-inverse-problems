/**
 * Mixer Module
 * Handles signal mixing using invertible mixing matrices
 */

import { MatrixOps, generateMixingMatrix } from './utils.js';

export class Mixer {
    constructor() {
        this.mixingMatrix = null;
        this.originalSources = null;
        this.mixedSignals = null;
        this.numSources = 0;
    }

    /**
     * Mix selected sources
     * @param {Array} selectedSounds - Array of sound objects with data
     * @returns {Object} Mixing result
     */
    mix(selectedSounds) {
        try {
            this.numSources = selectedSounds.length;
            
            // Construct source matrix S (n × N)
            // Each row is a source signal
            const signalLength = selectedSounds[0].data.length;
            
            // Ensure all signals have same length
            const minLength = Math.min(...selectedSounds.map(s => s.data.length));
            
            this.originalSources = MatrixOps.create(this.numSources, minLength);
            
            for (let i = 0; i < this.numSources; i++) {
                for (let j = 0; j < minLength; j++) {
                    this.originalSources[i][j] = selectedSounds[i].data[j];
                }
            }
            
            // Generate random mixing matrix A (n × n)
            this.mixingMatrix = generateMixingMatrix(this.numSources);
            
            console.log('Mixing Matrix A:');
            console.log(this.mixingMatrix);
            
            // Compute mixed signals X = A · S
            this.mixedSignals = MatrixOps.multiply(this.mixingMatrix, this.originalSources);
            
            // Convert to Float32Arrays for audio playback
            const mixedAudioData = [];
            for (let i = 0; i < this.numSources; i++) {
                mixedAudioData.push(new Float32Array(this.mixedSignals[i]));
            }
            
            return {
                mixedSignals: mixedAudioData,
                mixingMatrix: this.mixingMatrix,
                numSources: this.numSources,
                signalLength: minLength
            };
            
        } catch (error) {
            console.error('Mixing failed:', error);
            throw error;
        }
    }

    /**
     * Get mixing matrix
     * @returns {Float32Array[]}
     */
    getMixingMatrix() {
        return this.mixingMatrix;
    }

    /**
     * Get original sources
     * @returns {Float32Array[]}
     */
    getOriginalSources() {
        return this.originalSources;
    }

    /**
     * Get mixed signals
     * @returns {Float32Array[]}
     */
    getMixedSignals() {
        return this.mixedSignals;
    }

    /**
     * Get number of sources
     * @returns {number}
     */
    getNumSources() {
        return this.numSources;
    }

    /**
     * Reset mixer state
     */
    reset() {
        this.mixingMatrix = null;
        this.originalSources = null;
        this.mixedSignals = null;
        this.numSources = 0;
    }

    /**
     * Verify mixing matrix invertibility
     * @returns {boolean}
     */
    verifyInvertibility() {
        if (!this.mixingMatrix) return false;
        
        try {
            const det = MatrixOps.determinant(this.mixingMatrix);
            return Math.abs(det) > 1e-6;
        } catch (error) {
            return false;
        }
    }
}