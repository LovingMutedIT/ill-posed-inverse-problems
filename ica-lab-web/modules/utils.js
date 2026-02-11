/**
 * Utility functions for ICA Laboratory
 * Provides mathematical operations and helper functions
 */

/**
 * Matrix operations using Float32Array for performance
 */
export const MatrixOps = {
    /**
     * Create a matrix (2D array representation)
     * @param {number} rows 
     * @param {number} cols 
     * @returns {Float32Array[]}
     */
    create(rows, cols) {
        const matrix = new Array(rows);
        for (let i = 0; i < rows; i++) {
            matrix[i] = new Float32Array(cols);
        }
        return matrix;
    },

    /**
     * Matrix multiplication: C = A * B
     * @param {Float32Array[]} A - Matrix of size m x n
     * @param {Float32Array[]} B - Matrix of size n x p
     * @returns {Float32Array[]} - Matrix of size m x p
     */
    multiply(A, B) {
        const m = A.length;
        const n = B.length;
        const p = B[0].length;
        
        const C = this.create(m, p);
        
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < p; j++) {
                let sum = 0;
                for (let k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        
        return C;
    },

    /**
     * Matrix transpose
     * @param {Float32Array[]} A 
     * @returns {Float32Array[]}
     */
    transpose(A) {
        const rows = A.length;
        const cols = A[0].length;
        const AT = this.create(cols, rows);
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                AT[j][i] = A[i][j];
            }
        }
        
        return AT;
    },

    /**
     * Element-wise matrix addition
     * @param {Float32Array[]} A 
     * @param {Float32Array[]} B 
     * @returns {Float32Array[]}
     */
    add(A, B) {
        const rows = A.length;
        const cols = A[0].length;
        const C = this.create(rows, cols);
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
        
        return C;
    },

    /**
     * Element-wise matrix subtraction
     * @param {Float32Array[]} A 
     * @param {Float32Array[]} B 
     * @returns {Float32Array[]}
     */
    subtract(A, B) {
        const rows = A.length;
        const cols = A[0].length;
        const C = this.create(rows, cols);
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        
        return C;
    },

    /**
     * Scalar multiplication
     * @param {Float32Array[]} A 
     * @param {number} scalar 
     * @returns {Float32Array[]}
     */
    scale(A, scalar) {
        const rows = A.length;
        const cols = A[0].length;
        const C = this.create(rows, cols);
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                C[i][j] = A[i][j] * scalar;
            }
        }
        
        return C;
    },

    /**
     * Create identity matrix
     * @param {number} n 
     * @returns {Float32Array[]}
     */
    identity(n) {
        const I = this.create(n, n);
        for (let i = 0; i < n; i++) {
            I[i][i] = 1;
        }
        return I;
    },

    /**
     * Compute matrix inverse using Gaussian elimination
     * @param {Float32Array[]} A 
     * @returns {Float32Array[]}
     */
    inverse(A) {
        const n = A.length;
        const augmented = this.create(n, 2 * n);
        
        // Create augmented matrix [A | I]
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                augmented[i][j] = A[i][j];
                augmented[i][j + n] = (i === j) ? 1 : 0;
            }
        }
        
        // Gaussian elimination with partial pivoting
        for (let i = 0; i < n; i++) {
            // Find pivot
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            // Swap rows
            [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
            
            // Scale pivot row
            const pivot = augmented[i][i];
            if (Math.abs(pivot) < 1e-10) {
                throw new Error('Matrix is singular and cannot be inverted');
            }
            
            for (let j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            // Eliminate column
            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    const factor = augmented[k][i];
                    for (let j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Extract inverse from right half
        const inv = this.create(n, n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                inv[i][j] = augmented[i][j + n];
            }
        }
        
        return inv;
    },

    /**
     * Compute matrix determinant
     * @param {Float32Array[]} A 
     * @returns {number}
     */
    determinant(A) {
        const n = A.length;
        
        if (n === 1) return A[0][0];
        if (n === 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];
        
        // LU decomposition approach for efficiency
        const L = this.create(n, n);
        const U = this.create(n, n);
        
        for (let i = 0; i < n; i++) {
            // Upper triangular
            for (let k = i; k < n; k++) {
                let sum = 0;
                for (let j = 0; j < i; j++) {
                    sum += L[i][j] * U[j][k];
                }
                U[i][k] = A[i][k] - sum;
            }
            
            // Lower triangular
            for (let k = i; k < n; k++) {
                if (i === k) {
                    L[i][i] = 1;
                } else {
                    let sum = 0;
                    for (let j = 0; j < i; j++) {
                        sum += L[k][j] * U[j][i];
                    }
                    L[k][i] = (A[k][i] - sum) / U[i][i];
                }
            }
        }
        
        // Determinant is product of diagonal of U
        let det = 1;
        for (let i = 0; i < n; i++) {
            det *= U[i][i];
        }
        
        return det;
    },

    /**
     * Compute eigenvalues and eigenvectors using power iteration and deflation
     * For symmetric matrices only
     * @param {Float32Array[]} A - Symmetric matrix
     * @returns {{values: Float32Array, vectors: Float32Array[]}}
     */
    eigen(A) {
        const n = A.length;
        const maxIterations = 1000;
        const tolerance = 1e-8;
        
        const eigenvalues = new Float32Array(n);
        const eigenvectors = this.create(n, n);
        
        let B = this.create(n, n);
        // Copy A to B
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                B[i][j] = A[i][j];
            }
        }
        
        // Find each eigenvalue/eigenvector pair
        for (let k = 0; k < n; k++) {
            // Initialize random vector
            let v = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                v[i] = Math.random();
            }
            
            // Normalize
            let norm = 0;
            for (let i = 0; i < n; i++) {
                norm += v[i] * v[i];
            }
            norm = Math.sqrt(norm);
            for (let i = 0; i < n; i++) {
                v[i] /= norm;
            }
            
            // Power iteration
            let eigenvalue = 0;
            for (let iter = 0; iter < maxIterations; iter++) {
                const Bv = new Float32Array(n);
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        Bv[i] += B[i][j] * v[j];
                    }
                }
                
                eigenvalue = 0;
                for (let i = 0; i < n; i++) {
                    eigenvalue += v[i] * Bv[i];
                }
                
                norm = 0;
                for (let i = 0; i < n; i++) {
                    norm += Bv[i] * Bv[i];
                }
                norm = Math.sqrt(norm);
                
                if (norm < tolerance) break;
                
                let diff = 0;
                for (let i = 0; i < n; i++) {
                    const newV = Bv[i] / norm;
                    diff += Math.abs(newV - v[i]);
                    v[i] = newV;
                }
                
                if (diff < tolerance) break;
            }
            
            eigenvalues[k] = eigenvalue;
            for (let i = 0; i < n; i++) {
                eigenvectors[i][k] = v[i];
            }
            
            // Deflate B
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    B[i][j] -= eigenvalue * v[i] * v[j];
                }
            }
        }
        
        return { values: eigenvalues, vectors: eigenvectors };
    }
};

/**
 * Statistical operations
 */
export const Stats = {
    /**
     * Compute mean of array
     * @param {Float32Array} arr 
     * @returns {number}
     */
    mean(arr) {
        let sum = 0;
        for (let i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        return sum / arr.length;
    },

    /**
     * Compute standard deviation
     * @param {Float32Array} arr 
     * @returns {number}
     */
    std(arr) {
        const mu = this.mean(arr);
        let sum = 0;
        for (let i = 0; i < arr.length; i++) {
            const diff = arr[i] - mu;
            sum += diff * diff;
        }
        return Math.sqrt(sum / arr.length);
    },

    /**
     * Compute kurtosis (excess kurtosis)
     * @param {Float32Array} arr 
     * @returns {number}
     */
    kurtosis(arr) {
        const mu = this.mean(arr);
        const sigma = this.std(arr);
        
        if (sigma < 1e-10) return 0;
        
        let sum = 0;
        for (let i = 0; i < arr.length; i++) {
            const z = (arr[i] - mu) / sigma;
            sum += z * z * z * z;
        }
        
        return (sum / arr.length) - 3;
    },

    /**
     * Compute skewness
     * @param {Float32Array} arr 
     * @returns {number}
     */
    skewness(arr) {
        const mu = this.mean(arr);
        const sigma = this.std(arr);
        
        if (sigma < 1e-10) return 0;
        
        let sum = 0;
        for (let i = 0; i < arr.length; i++) {
            const z = (arr[i] - mu) / sigma;
            sum += z * z * z;
        }
        
        return sum / arr.length;
    },

    /**
     * Normalize array to zero mean and unit variance
     * @param {Float32Array} arr 
     * @returns {Float32Array}
     */
    normalize(arr) {
        const mu = this.mean(arr);
        const sigma = this.std(arr);
        
        const normalized = new Float32Array(arr.length);
        
        if (sigma < 1e-10) {
            return normalized; // Return zeros if no variance
        }
        
        for (let i = 0; i < arr.length; i++) {
            normalized[i] = (arr[i] - mu) / sigma;
        }
        
        return normalized;
    },

    /**
     * Compute correlation between two arrays
     * @param {Float32Array} x 
     * @param {Float32Array} y 
     * @returns {number}
     */
    correlation(x, y) {
        if (x.length !== y.length) {
            throw new Error('Arrays must have same length');
        }
        
        const n = x.length;
        const mx = this.mean(x);
        const my = this.mean(y);
        
        let num = 0, dx = 0, dy = 0;
        for (let i = 0; i < n; i++) {
            const diffX = x[i] - mx;
            const diffY = y[i] - my;
            num += diffX * diffY;
            dx += diffX * diffX;
            dy += diffY * diffY;
        }
        
        const denom = Math.sqrt(dx * dy);
        return denom < 1e-10 ? 0 : num / denom;
    }
};

/**
 * Audio utilities
 */
export const AudioUtils = {
    /**
     * Convert audio buffer to mono
     * @param {AudioBuffer} buffer 
     * @returns {Float32Array}
     */
    bufferToMono(buffer) {
        if (buffer.numberOfChannels === 1) {
            return new Float32Array(buffer.getChannelData(0));
        }
        
        const left = buffer.getChannelData(0);
        const right = buffer.getChannelData(1);
        const mono = new Float32Array(left.length);
        
        for (let i = 0; i < left.length; i++) {
            mono[i] = (left[i] + right[i]) / 2;
        }
        
        return mono;
    },

    /**
     * Resample audio data
     * @param {Float32Array} data 
     * @param {number} fromRate 
     * @param {number} toRate 
     * @returns {Float32Array}
     */
    resample(data, fromRate, toRate) {
        if (fromRate === toRate) {
            return new Float32Array(data);
        }
        
        const ratio = fromRate / toRate;
        const newLength = Math.floor(data.length / ratio);
        const resampled = new Float32Array(newLength);
        
        for (let i = 0; i < newLength; i++) {
            const srcIndex = i * ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, data.length - 1);
            const fraction = srcIndex - srcIndexFloor;
            
            resampled[i] = data[srcIndexFloor] * (1 - fraction) + 
                          data[srcIndexCeil] * fraction;
        }
        
        return resampled;
    },

    /**
     * Create audio buffer from Float32Array
     * @param {AudioContext} ctx 
     * @param {Float32Array} data 
     * @param {number} sampleRate 
     * @returns {AudioBuffer}
     */
    createBuffer(ctx, data, sampleRate) {
        const buffer = ctx.createBuffer(1, data.length, sampleRate);
        buffer.copyToChannel(data, 0);
        return buffer;
    },

    /**
     * Format duration in seconds to MM:SS
     * @param {number} seconds 
     * @returns {string}
     */
    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
};

/**
 * Generate random mixing matrix
 * @param {number} n - Matrix dimension
 * @returns {Float32Array[]}
 */
export function generateMixingMatrix(n) {
    const A = MatrixOps.create(n, n);
    
    // Generate random matrix
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            A[i][j] = (Math.random() - 0.5) * 2;
        }
    }
    
    // Ensure invertibility
    let det = MatrixOps.determinant(A);
    let attempts = 0;
    
    while (Math.abs(det) < 0.1 && attempts < 10) {
        // Add small diagonal regularization
        for (let i = 0; i < n; i++) {
            A[i][i] += 0.1;
        }
        det = MatrixOps.determinant(A);
        attempts++;
    }
    
    if (Math.abs(det) < 0.01) {
        throw new Error('Failed to generate invertible mixing matrix');
    }
    
    return A;
}

/**
 * Show notification message
 * @param {string} message 
 * @param {string} type - 'error', 'warning', or 'success'
 * @param {number} duration - Duration in ms
 */
export function showNotification(message, type = 'info', duration = 3000) {
    const container = document.getElementById('validationMessages');
    
    const notification = document.createElement('div');
    notification.className = `validation-message ${type}`;
    notification.textContent = message;
    
    container.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideIn var(--transition-base) ease-out reverse';
        setTimeout(() => notification.remove(), 250);
    }, duration);
}

/**
 * Generate unique ID
 * @returns {string}
 */
export function generateId() {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}