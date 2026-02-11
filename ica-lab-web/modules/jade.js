/**
 * JADE Module
 * Joint Approximate Diagonalization of Eigenmatrices
 * Fourth-order cumulant-based ICA algorithm
 */

import { MatrixOps } from './utils.js';

export class JADE {
    constructor(options = {}) {
        this.maxIterations = options.maxIterations || 100;
        this.tolerance = options.tolerance || 1e-6;
        this.unmixingMatrix = null;
        this.sources = null;
    }

    /**
     * JADE algorithm
     * @param {Float32Array[]} X - Whitened data matrix (n × N)
     * @param {Function} progressCallback - Optional progress callback
     * @returns {Object} Result containing unmixing matrix and sources
     */
    fit(X, progressCallback = null) {
        const n = X.length; // Number of sources
        const N = X[0].length; // Number of samples
        
        console.log(`JADE: Processing ${n} sources with ${N} samples`);
        
        if (n > 6) {
            console.warn('JADE with >6 sources may be computationally expensive');
        }
        
        // Step 1: Compute fourth-order cumulant matrices
        console.log('Computing cumulant matrices...');
        const cumulantMatrices = this.computeCumulantMatrices(X);
        
        if (progressCallback) {
            progressCallback({
                step: 'cumulants',
                progress: 0.3
            });
        }
        
        // Step 2: Joint approximate diagonalization
        console.log('Performing joint diagonalization...');
        const V = this.jointDiagonalization(cumulantMatrices, progressCallback);
        
        if (progressCallback) {
            progressCallback({
                step: 'diagonalization',
                progress: 0.9
            });
        }
        
        this.unmixingMatrix = V;
        
        // Step 3: Recover sources S = V · X
        this.sources = MatrixOps.multiply(V, X);
        
        console.log('JADE completed');
        
        return {
            unmixingMatrix: V,
            sources: this.sources,
            cumulantMatrices: cumulantMatrices.length
        };
    }

    /**
     * Compute fourth-order cumulant matrices
     * @param {Float32Array[]} X - Whitened data (n × N)
     * @returns {Float32Array[][]} Array of cumulant matrices
     */
    computeCumulantMatrices(X) {
        const n = X.length;
        const N = X[0].length;
        
        const cumulantMatrices = [];
        
        // For whitened data, we compute cumulant matrices Q_i
        // where Q_i = E[x_i^2 xx^T] - I - 2e_i e_i^T
        
        for (let i = 0; i < n; i++) {
            const Q = MatrixOps.create(n, n);
            
            // Compute E[x_i^2 xx^T]
            for (let j = 0; j < n; j++) {
                for (let k = 0; k < n; k++) {
                    let sum = 0;
                    for (let t = 0; t < N; t++) {
                        sum += X[i][t] * X[i][t] * X[j][t] * X[k][t];
                    }
                    Q[j][k] = sum / N;
                }
            }
            
            // Subtract identity
            for (let j = 0; j < n; j++) {
                Q[j][j] -= 1;
            }
            
            // Subtract 2e_i e_i^T
            Q[i][i] -= 2;
            
            cumulantMatrices.push(Q);
        }
        
        // Also compute off-diagonal cumulant matrices
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const Q = MatrixOps.create(n, n);
                
                // Compute E[x_i x_j xx^T]
                for (let k = 0; k < n; k++) {
                    for (let l = 0; l < n; l++) {
                        let sum = 0;
                        for (let t = 0; t < N; t++) {
                            sum += X[i][t] * X[j][t] * X[k][t] * X[l][t];
                        }
                        Q[k][l] = sum / N;
                    }
                }
                
                // For whitened data with zero mean, 
                // subtract E[x_i x_j] E[x_k x_l] terms
                // But since data is whitened, E[x_i x_j] = δ_ij
                if (i === j) {
                    Q[i][j] -= 1;
                }
                
                cumulantMatrices.push(Q);
            }
        }
        
        return cumulantMatrices;
    }

    /**
     * Joint approximate diagonalization using Givens rotations
     * @param {Float32Array[][]} matrices - Array of matrices to jointly diagonalize
     * @param {Function} progressCallback 
     * @returns {Float32Array[]}
     */
    jointDiagonalization(matrices, progressCallback = null) {
        const n = matrices[0].length;
        
        // Initialize rotation matrix as identity
        let V = MatrixOps.identity(n);
        
        let converged = false;
        let iteration = 0;
        
        // Transform matrices array
        const transformedMatrices = matrices.map(M => {
            const result = MatrixOps.create(n, n);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    result[i][j] = M[i][j];
                }
            }
            return result;
        });
        
        while (!converged && iteration < this.maxIterations) {
            let maxChange = 0;
            
            // Sweep through all pairs (i,j) with i < j
            for (let i = 0; i < n - 1; i++) {
                for (let j = i + 1; j < n; j++) {
                    // Compute optimal Givens rotation angle
                    const angle = this.computeGivensAngle(transformedMatrices, i, j);
                    
                    if (Math.abs(angle) > this.tolerance) {
                        // Apply Givens rotation
                        const G = this.createGivensRotation(n, i, j, angle);
                        const GT = MatrixOps.transpose(G);
                        
                        // Update V
                        V = MatrixOps.multiply(V, G);
                        
                        // Update all cumulant matrices: M' = G^T M G
                        for (let m = 0; m < transformedMatrices.length; m++) {
                            const temp = MatrixOps.multiply(GT, transformedMatrices[m]);
                            transformedMatrices[m] = MatrixOps.multiply(temp, G);
                        }
                        
                        maxChange = Math.max(maxChange, Math.abs(angle));
                    }
                }
            }
            
            iteration++;
            
            if (progressCallback && iteration % 5 === 0) {
                progressCallback({
                    step: 'diagonalization',
                    iteration,
                    maxChange,
                    progress: 0.3 + 0.6 * Math.min(1, iteration / this.maxIterations)
                });
            }
            
            if (maxChange < this.tolerance) {
                converged = true;
            }
        }
        
        console.log(`JADE joint diagonalization converged in ${iteration} iterations`);
        
        return V;
    }

    /**
     * Compute optimal Givens rotation angle for pair (i,j)
     * @param {Float32Array[][]} matrices 
     * @param {number} i 
     * @param {number} j 
     * @returns {number} Rotation angle
     */
    computeGivensAngle(matrices, i, j) {
        let h = 0, g = 0;
        
        // Sum over all matrices
        for (const M of matrices) {
            const a = M[i][i] - M[j][j];
            const b = M[i][j] + M[j][i];
            
            h += a * a + b * b;
            g += a * b;
        }
        
        if (Math.abs(h) < 1e-10) {
            return 0;
        }
        
        // Compute angle: θ = 0.25 * atan2(2g, h)
        const angle = 0.25 * Math.atan2(2 * g, h);
        
        return angle;
    }

    /**
     * Create Givens rotation matrix
     * @param {number} n - Matrix size
     * @param {number} i - First index
     * @param {number} j - Second index
     * @param {number} angle - Rotation angle
     * @returns {Float32Array[]}
     */
    createGivensRotation(n, i, j, angle) {
        const G = MatrixOps.identity(n);
        
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        
        G[i][i] = c;
        G[j][j] = c;
        G[i][j] = -s;
        G[j][i] = s;
        
        return G;
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