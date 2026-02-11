/**
 * Validator Module
 * Non-Gaussian validation for ICA sources
 */

import { Stats } from './utils.js';

export class Validator {
    constructor(settings) {
        this.thresholds = settings.validationThresholds;
    }

    /**
     * Validate if sources are suitable for ICA
     * @param {Array} selectedSounds - Array of sound objects
     * @returns {Object} Validation result
     */
    validate(selectedSounds) {
        const result = {
            isValid: true,
            errors: [],
            warnings: [],
            details: []
        };

        // Check minimum number of sources
        if (selectedSounds.length < 2) {
            result.isValid = false;
            result.errors.push('At least 2 sources required for ICA');
            return result;
        }

        // Validate each source
        selectedSounds.forEach((sound, index) => {
            const validation = this.validateSource(sound);
            
            result.details.push({
                soundId: sound.id,
                soundName: sound.name,
                kurtosis: sound.kurtosis,
                level: validation.level,
                isValid: validation.isValid
            });

            if (!validation.isValid) {
                result.isValid = false;
                result.errors.push(
                    `${sound.name}: Nearly Gaussian (|kurtosis| = ${Math.abs(sound.kurtosis).toFixed(3)})`
                );
            } else if (validation.level === 'weakly') {
                result.warnings.push(
                    `${sound.name}: Weakly non-Gaussian (|kurtosis| = ${Math.abs(sound.kurtosis).toFixed(3)})`
                );
            }
        });

        return result;
    }

    /**
     * Validate individual source
     * @param {Object} sound 
     * @returns {Object}
     */
    validateSource(sound) {
        const absKurtosis = Math.abs(sound.kurtosis);
        
        let level, isValid;
        
        if (absKurtosis < this.thresholds.nearlyGaussian) {
            level = 'nearly-gaussian';
            isValid = false;
        } else if (absKurtosis < this.thresholds.weaklyNonGaussian) {
            level = 'weakly';
            isValid = true;
        } else {
            level = 'strongly';
            isValid = true;
        }

        return { level, isValid };
    }

    /**
     * Compute advanced non-Gaussianity measure using negentropy
     * @param {Float32Array} data 
     * @returns {number}
     */
    computeNegentropy(data) {
        const normalized = Stats.normalize(data);
        const n = normalized.length;
        
        // Approximate negentropy using log-cosh nonlinearity
        // J(y) ≈ [E{G(y)} - E{G(v)}]^2
        // where G(u) = log(cosh(u)) and v ~ N(0,1)
        
        let sumG = 0;
        for (let i = 0; i < n; i++) {
            // G(u) = log(cosh(u))
            const u = normalized[i];
            sumG += Math.log(Math.cosh(u));
        }
        
        const EG = sumG / n;
        
        // For Gaussian: E{G(v)} ≈ 0.3747 for G(u) = log(cosh(u))
        const EGaussian = 0.3747;
        
        const negentropy = Math.pow(EG - EGaussian, 2);
        
        return negentropy;
    }

    /**
     * Get validation message for UI
     * @param {Object} validationResult 
     * @returns {string}
     */
    getValidationMessage(validationResult) {
        if (!validationResult.isValid) {
            return validationResult.errors.join('; ');
        }
        
        if (validationResult.warnings.length > 0) {
            return validationResult.warnings.join('; ');
        }
        
        return 'All sources validated successfully';
    }
}