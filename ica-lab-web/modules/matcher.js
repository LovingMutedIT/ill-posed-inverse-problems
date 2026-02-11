/**
 * Matcher Module
 * Matches recovered sources to original sources using correlation
 */

import { Stats } from './utils.js';

export class SourceMatcher {
    constructor() {
        this.matchResults = null;
    }

    /**
     * Match recovered sources to original sources
     * @param {Float32Array[]} originalSources - Original source signals
     * @param {Float32Array[]} recoveredSources - ICA-recovered sources
     * @param {Array} originalNames - Names of original sources
     * @returns {Array} Match results with labels and confidence
     */
    match(originalSources, recoveredSources, originalNames) {
        const n = recoveredSources.length;
        const results = [];
        
        // Track which original sources have been matched
        const matched = new Set();
        
        // For each recovered source, find best matching original source
        for (let i = 0; i < n; i++) {
            const recovered = recoveredSources[i];
            
            let bestMatch = -1;
            let bestCorrelation = -1;
            let bestSign = 1;
            
            // Compare with all original sources
            for (let j = 0; j < n; j++) {
                if (matched.has(j)) continue; // Skip already matched
                
                const original = originalSources[j];
                
                // Compute correlation (handles sign ambiguity)
                const corr = Stats.correlation(recovered, original);
                const absCorr = Math.abs(corr);
                
                if (absCorr > bestCorrelation) {
                    bestCorrelation = absCorr;
                    bestMatch = j;
                    bestSign = corr > 0 ? 1 : -1;
                }
            }
            
            // Mark as matched
            matched.add(bestMatch);
            
            // Compute quality metrics
            const confidence = bestCorrelation * 100;
            const quality = this.getQualityLevel(bestCorrelation);
            
            results.push({
                recoveredIndex: i,
                matchedIndex: bestMatch,
                matchedName: originalNames[bestMatch],
                correlation: bestCorrelation,
                sign: bestSign,
                confidence: confidence,
                quality: quality
            });
        }
        
        this.matchResults = results;
        return results;
    }

    /**
     * Get quality level based on correlation
     * @param {number} correlation - Absolute correlation value
     * @returns {string}
     */
    getQualityLevel(correlation) {
        if (correlation >= 0.9) {
            return 'excellent';
        } else if (correlation >= 0.7) {
            return 'good';
        } else if (correlation >= 0.5) {
            return 'fair';
        } else {
            return 'poor';
        }
    }

    /**
     * Get match results
     * @returns {Array}
     */
    getMatchResults() {
        return this.matchResults;
    }

    /**
     * Get average separation quality
     * @returns {number}
     */
    getAverageSeparationQuality() {
        if (!this.matchResults) return 0;
        
        const avgCorr = this.matchResults.reduce((sum, r) => sum + r.correlation, 0) / this.matchResults.length;
        return avgCorr;
    }

    /**
     * Apply sign correction to recovered sources
     * @param {Float32Array[]} recoveredSources 
     * @returns {Float32Array[]}
     */
    applySignCorrection(recoveredSources) {
        if (!this.matchResults) return recoveredSources;
        
        const corrected = [];
        
        for (const match of this.matchResults) {
            const source = new Float32Array(recoveredSources[match.recoveredIndex]);
            
            if (match.sign < 0) {
                // Flip sign
                for (let i = 0; i < source.length; i++) {
                    source[i] = -source[i];
                }
            }
            
            corrected.push(source);
        }
        
        return corrected;
    }

    /**
     * Reorder recovered sources to match original order
     * @param {Float32Array[]} recoveredSources 
     * @returns {Float32Array[]}
     */
    reorderSources(recoveredSources) {
        if (!this.matchResults) return recoveredSources;
        
        const reordered = new Array(recoveredSources.length);
        
        for (const match of this.matchResults) {
            reordered[match.matchedIndex] = recoveredSources[match.recoveredIndex];
        }
        
        return reordered;
    }

    /**
     * Reset matcher state
     */
    reset() {
        this.matchResults = null;
    }
}