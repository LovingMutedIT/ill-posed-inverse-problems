/**
 * ICA Controller Module
 * Coordinates ICA execution with preprocessing
 */

import { Preprocessor } from './preprocess.js';
import { FastICA } from './fastica.js';
import { JADE } from './jade.js';

export class ICAController {
    constructor(settings) {
        this.settings = settings;
        this.preprocessor = new Preprocessor();
        this.algorithm = null;
        this.selectedAlgorithm = 'fastica';
        this.worker = null;
    }

    /**
     * Set algorithm type
     * @param {string} algorithm - 'fastica' or 'jade'
     */
    setAlgorithm(algorithm) {
        this.selectedAlgorithm = algorithm;
    }

    /**
     * Check if JADE is appropriate for given number of sources
     * @param {number} numSources 
     * @returns {Object}
     */
    checkJADEFeasibility(numSources) {
        const maxSources = this.settings.icaSettings.jadeMaxSources;
        
        return {
            feasible: numSources <= maxSources,
            warning: numSources > maxSources,
            message: numSources > maxSources 
                ? `JADE with ${numSources} sources may be very slow. Consider using FastICA or reducing sources to ≤${maxSources}.`
                : null
        };
    }

    /**
     * Separate sources using selected ICA algorithm
     * @param {Float32Array[]} mixedSignals - Mixed signals matrix
     * @param {Function} progressCallback - Progress callback
     * @returns {Promise<Object>}
     */
    async separate(mixedSignals, progressCallback = null) {
        try {
            const n = mixedSignals.length;
            
            // Check JADE feasibility
            if (this.selectedAlgorithm === 'jade') {
                const feasibility = this.checkJADEFeasibility(n);
                if (feasibility.warning && progressCallback) {
                    progressCallback({
                        type: 'warning',
                        message: feasibility.message
                    });
                }
            }
            
            // Step 1: Preprocess (center and whiten)
            if (progressCallback) {
                progressCallback({
                    step: 'preprocessing',
                    progress: 0.1,
                    message: 'Centering and whitening data...'
                });
            }
            
            const whitened = this.preprocessor.preprocess(mixedSignals);
            
            // Step 2: Run ICA algorithm
            if (progressCallback) {
                progressCallback({
                    step: 'ica',
                    progress: 0.2,
                    message: `Running ${this.selectedAlgorithm.toUpperCase()}...`
                });
            }
            
            let result;
            
            if (this.selectedAlgorithm === 'fastica') {
                this.algorithm = new FastICA({
                    maxIterations: this.settings.icaSettings.maxIterations,
                    tolerance: this.settings.icaSettings.tolerance
                });
                
                result = this.algorithm.fit(whitened, (icaProgress) => {
                    if (progressCallback) {
                        progressCallback({
                            step: 'ica',
                            progress: 0.2 + 0.7 * (icaProgress.iteration / this.settings.icaSettings.maxIterations),
                            message: `FastICA iteration ${icaProgress.iteration}`,
                            ...icaProgress
                        });
                    }
                });
            } else if (this.selectedAlgorithm === 'jade') {
                this.algorithm = new JADE({
                    maxIterations: this.settings.icaSettings.maxIterations,
                    tolerance: this.settings.icaSettings.tolerance
                });
                
                result = this.algorithm.fit(whitened, (jadeProgress) => {
                    if (progressCallback) {
                        progressCallback({
                            step: jadeProgress.step || 'ica',
                            progress: jadeProgress.progress || 0.5,
                            message: `JADE ${jadeProgress.step || 'processing'}...`,
                            ...jadeProgress
                        });
                    }
                });
            }
            
            // Step 3: Convert to Float32Arrays for audio
            const separatedSources = [];
            for (let i = 0; i < result.sources.length; i++) {
                separatedSources.push(new Float32Array(result.sources[i]));
            }
            
            if (progressCallback) {
                progressCallback({
                    step: 'complete',
                    progress: 1.0,
                    message: 'Separation complete'
                });
            }
            
            return {
                sources: separatedSources,
                unmixingMatrix: result.unmixingMatrix,
                preprocessor: this.preprocessor,
                algorithm: this.selectedAlgorithm,
                iterations: result.iterations,
                converged: result.converged
            };
            
        } catch (error) {
            console.error('ICA separation failed:', error);
            throw error;
        }
    }

    /**
     * Separate using Web Worker (for heavy computation)
     * @param {Float32Array[]} mixedSignals 
     * @param {Function} progressCallback 
     * @returns {Promise<Object>}
     */
    async separateWithWorker(mixedSignals, progressCallback = null) {
        return new Promise((resolve, reject) => {
            // Terminate existing worker if any
            if (this.worker) {
                this.worker.terminate();
            }
            
            // Create new worker
            this.worker = new Worker('workers/icaWorker.js', { type: 'module' });
            
            // Handle messages from worker
            this.worker.onmessage = (e) => {
                const { type, data } = e.data;
                
                if (type === 'progress' && progressCallback) {
                    progressCallback(data);
                } else if (type === 'complete') {
                    // Reconstruct Float32Arrays from transferred data
                    const sources = data.sources.map(s => new Float32Array(s));
                    
                    resolve({
                        sources,
                        unmixingMatrix: data.unmixingMatrix,
                        algorithm: data.algorithm,
                        iterations: data.iterations,
                        converged: data.converged
                    });
                    
                    this.worker.terminate();
                    this.worker = null;
                } else if (type === 'error') {
                    reject(new Error(data.message));
                    this.worker.terminate();
                    this.worker = null;
                }
            };
            
            this.worker.onerror = (error) => {
                reject(error);
                this.worker.terminate();
                this.worker = null;
            };
            
            // Send data to worker
            this.worker.postMessage({
                type: 'separate',
                data: {
                    mixedSignals: mixedSignals.map(s => Array.from(s)),
                    algorithm: this.selectedAlgorithm,
                    settings: this.settings
                }
            });
        });
    }

    /**
     * Reset controller state
     */
    reset() {
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
        
        this.preprocessor.reset();
        
        if (this.algorithm) {
            this.algorithm.reset();
        }
        
        this.selectedAlgorithm = 'fastica';
    }

    /**
     * Get preprocessor
     * @returns {Preprocessor}
     */
    getPreprocessor() {
        return this.preprocessor;
    }

    /**
     * Get algorithm instance
     * @returns {FastICA|JADE}
     */
    getAlgorithm() {
        return this.algorithm;
    }
}