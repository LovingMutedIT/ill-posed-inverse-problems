/**
 * FastICA Module
 * Symmetric FastICA algorithm implementation
 * Uses negentropy maximization with tanh nonlinearity
 */

import { MatrixOps } from './utils.js';

export class FastICA {
    constructor(options = {}) {
        this.maxIterations = options.maxIterations || 1000;
        this.tolerance = options.tolerance || 1e-6;
        this.unmixingMatrix = null;
        this.sources = null;
    }

    /**
     * Symmetric FastICA algorithm
     * @param {Float32Array[]} X - Whitened data matrix (n × N)
     * @param {Function} progressCallback - Optional progress callback
     * @returns {Object} Result containing unmixing matrix and sources
     */
    fit(X, progressCallback = null) {
        const n = X.length; // Number of sources
        const N = X[0].length; // Number of samples
        
        console.log(`FastICA: Processing ${n} sources with ${N} samples`);
        
        // Initialize unmixing matrix W with random orthogonal matrix
        let W = this.initializeOrthogonal(n);
        
        let converged = false;
        let iteration = 0;
        
        while (!converged && iteration < this.maxIterations) {
            // Store previous W for convergence check
            const WOld = MatrixOps.create(n, n);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    WOld[i][j] = W[i][j];
                }
            }
            
            // Compute WX for all components
            const WX = MatrixOps.multiply(W, X);
            
            // Update rule: W_new = E[g(WX)X^T] - E[g'(WX)]W
            // Using g(u) = tanh(u)
            
            // Compute E[g(WX)X^T]
            const gWX_XT = MatrixOps.create(n, n);
            
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    let sum = 0;
                    for (let k = 0; k < N; k++) {
                        const gVal = Math.tanh(WX[i][k]);
                        sum += gVal * X[j][k];
                    }
                    gWX_XT[i][j] = sum / N;
                }
            }
            
            // Compute E[g'(WX)]
            const gPrimeWX = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                let sum = 0;
                for (let k = 0; k < N; k++) {
                    const tanhVal = Math.tanh(WX[i][k]);
                    // Derivative: g'(u) = 1 - tanh^2(u)
                    sum += (1 - tanhVal * tanhVal);
                }
                gPrimeWX[i] = sum / N;
            }
            
            // Compute diagonal matrix from g'
            const gPrimeDiag = MatrixOps.create(n, n);
            for (let i = 0; i < n; i++) {
                gPrimeDiag[i][i] = gPrimeWX[i];
            }
            
            // W_new = E[g(WX)X^T] - E[g'(WX)]W
            const gPrimeW = MatrixOps.multiply(gPrimeDiag, W);
            const WNew = MatrixOps.subtract(gWX_XT, gPrimeW);
            
            // Symmetric decorrelation: W = (W W^T)^{-1/2} W
            W = this.symmetricDecorrelation(WNew);
            
            // Check convergence
            let maxDiff = 0;
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    const diff = Math.abs(Math.abs(W[i][j]) - Math.abs(WOld[i][j]));
                    maxDiff = Math.max(maxDiff, diff);
                }
            }
            
            iteration++;
            
            if (progressCallback && iteration % 10 === 0) {
                progressCallback({
                    iteration,
                    maxDiff,
                    converged: maxDiff < this.tolerance
                });
            }
            
            if (maxDiff < this.tolerance) {
                converged = true;
            }
        }
        
        console.log(`FastICA converged in ${iteration} iterations`);
        
        this.unmixingMatrix = W;
        
        // Recover sources: S = W · X
        this.sources = MatrixOps.multiply(W, X);
        
        return {
            unmixingMatrix: W,
            sources: this.sources,
            iterations: iteration,
            converged
        };
    }

    /**
     * Initialize random orthogonal matrix
     * @param {number} n - Matrix dimension
     * @returns {Float32Array[]}
     */
    initializeOrthogonal(n) {
        // Start with random matrix
        const W = MatrixOps.create(n, n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                W[i][j] = Math.random() - 0.5;
            }
        }
        
        // Orthogonalize using symmetric decorrelation
        return this.symmetricDecorrelation(W);
    }

    /**
     * Symmetric decorrelation: W = (W W^T)^{-1/2} W
     * @param {Float32Array[]} W 
     * @returns {Float32Array[]}
     */
    symmetricDecorrelation(W) {
        const n = W.length;
        
        // Compute W W^T
        const WT = MatrixOps.transpose(W);
        const WWT = MatrixOps.multiply(W, WT);
        
        // Eigenvalue decomposition of W W^T
        const eigen = MatrixOps.eigen(WWT);
        
        // Compute (W W^T)^{-1/2} = E D^{-1/2} E^T
        const DInvSqrt = MatrixOps.create(n, n);
        for (let i = 0; i < n; i++) {
            if (eigen.values[i] > 1e-10) {
                DInvSqrt[i][i] = 1 / Math.sqrt(eigen.values[i]);
            }
        }
        
        const ET = MatrixOps.transpose(eigen.vectors);
        const EDInvSqrt = MatrixOps.multiply(eigen.vectors, DInvSqrt);
        const WWTInvSqrt = MatrixOps.multiply(EDInvSqrt, ET);
        
        // Return (W W^T)^{-1/2} W
        return MatrixOps.multiply(WWTInvSqrt, W);
    }

    /**
     * Get unmixing matrix
     * @returns {Float32Array[]}
     */
    getUnmixingMatrix() {
        return this.unmixingMatrix;
    }

    /**
     * Get separated sources
     * @returns {Float32Array[]}
     */
    getSources() {
        return this.sources;
    }

    /**
     * Reset algorithm state
     */
    reset() {
        this.unmixingMatrix = null;
        this.sources = null;
    }
}