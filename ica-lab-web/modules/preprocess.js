/**
 * Preprocessing Module
 * Shared preprocessing pipeline for ICA algorithms
 * Implements centering and whitening
 */

import { MatrixOps, Stats } from './utils.js';

export class Preprocessor {
    constructor() {
        this.mean = null;
        this.whitenMatrix = null;
        this.dewhitenMatrix = null;
        this.centered = null;
        this.whitened = null;
    }

    /**
     * Center the data (subtract mean)
     * @param {Float32Array[]} X - Data matrix (n × N)
     * @returns {Float32Array[]}
     */
    center(X) {
        const n = X.length;
        const N = X[0].length;
        
        // Compute mean for each row
        this.mean = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = 0; j < N; j++) {
                sum += X[i][j];
            }
            this.mean[i] = sum / N;
        }
        
        // Subtract mean
        this.centered = MatrixOps.create(n, N);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < N; j++) {
                this.centered[i][j] = X[i][j] - this.mean[i];
            }
        }
        
        return this.centered;
    }

    /**
     * Whiten the data using eigenvalue decomposition
     * Transforms data to have identity covariance matrix
     * @param {Float32Array[]} X - Centered data matrix (n × N)
     * @returns {Float32Array[]}
     */
    whiten(X) {
        const n = X.length;
        const N = X[0].length;
        
        // Compute covariance matrix C = (1/N) X X^T
        const XT = MatrixOps.transpose(X);
        const XXT = MatrixOps.multiply(X, XT);
        const C = MatrixOps.scale(XXT, 1 / N);
        
        console.log('Covariance matrix computed');
        
        // Eigenvalue decomposition: C = E D E^T
        const eigen = MatrixOps.eigen(C);
        
        console.log('Eigenvalues:', eigen.values);
        
        // Create D^{-1/2}
        const DInvSqrt = MatrixOps.create(n, n);
        for (let i = 0; i < n; i++) {
            const eigenvalue = eigen.values[i];
            if (eigenvalue > 1e-10) {
                DInvSqrt[i][i] = 1 / Math.sqrt(eigenvalue);
            } else {
                // Handle near-zero eigenvalues
                DInvSqrt[i][i] = 0;
                console.warn(`Small eigenvalue detected: ${eigenvalue}`);
            }
        }
        
        // Whitening matrix: W = D^{-1/2} E^T
        const ET = MatrixOps.transpose(eigen.vectors);
        this.whitenMatrix = MatrixOps.multiply(DInvSqrt, ET);
        
        // Dewhitening matrix: W^{-1} = E D^{1/2}
        const DSqrt = MatrixOps.create(n, n);
        for (let i = 0; i < n; i++) {
            const eigenvalue = eigen.values[i];
            if (eigenvalue > 1e-10) {
                DSqrt[i][i] = Math.sqrt(eigenvalue);
            }
        }
        this.dewhitenMatrix = MatrixOps.multiply(eigen.vectors, DSqrt);
        
        // Apply whitening: X_white = W · X
        this.whitened = MatrixOps.multiply(this.whitenMatrix, X);
        
        // Verify whitening (covariance should be identity)
        if (typeof process !== 'undefined' && process.env.NODE_ENV === 'development') {
            const whiteXT = MatrixOps.transpose(this.whitened);
            const whiteCov = MatrixOps.multiply(this.whitened, whiteXT);
            const whiteCovScaled = MatrixOps.scale(whiteCov, 1 / N);
            console.log('Whitened covariance (should be ~I):', whiteCovScaled);
        }
        
        return this.whitened;
    }

    /**
     * Full preprocessing pipeline: center then whiten
     * @param {Float32Array[]} X - Raw data matrix
     * @returns {Float32Array[]}
     */
    preprocess(X) {
        const centered = this.center(X);
        const whitened = this.whiten(centered);
        return whitened;
    }

    /**
     * Get whitening matrix
     * @returns {Float32Array[]}
     */
    getWhitenMatrix() {
        return this.whitenMatrix;
    }

    /**
     * Get dewhitening matrix
     * @returns {Float32Array[]}
     */
    getDewhitenMatrix() {
        return this.dewhitenMatrix;
    }

    /**
     * Get centered data
     * @returns {Float32Array[]}
     */
    getCentered() {
        return this.centered;
    }

    /**
     * Get whitened data
     * @returns {Float32Array[]}
     */
    getWhitened() {
        return this.whitened;
    }

    /**
     * Get mean
     * @returns {Float32Array}
     */
    getMean() {
        return this.mean;
    }

    /**
     * Reset preprocessor state
     */
    reset() {
        this.mean = null;
        this.whitenMatrix = null;
        this.dewhitenMatrix = null;
        this.centered = null;
        this.whitened = null;
    }
}