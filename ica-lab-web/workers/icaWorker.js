/**
 * ICA Web Worker
 * Performs ICA computation off the main thread
 */

// Import modules - Note: Worker scripts need special handling for imports
// This will be loaded as a module worker

let Preprocessor, FastICA, JADE;

// Dynamic import for worker context
async function loadModules() {
    const preprocessModule = await import('../modules/preprocess.js');
    const fasticaModule = await import('../modules/fastica.js');
    const jadeModule = await import('../modules/jade.js');
    
    Preprocessor = preprocessModule.Preprocessor;
    FastICA = fasticaModule.FastICA;
    JADE = jadeModule.JADE;
}

// Initialize on worker startup
loadModules().catch(err => {
    self.postMessage({
        type: 'error',
        data: { message: `Failed to load modules: ${err.message}` }
    });
});

/**
 * Handle messages from main thread
 */
self.onmessage = async function(e) {
    const { type, data } = e.data;
    
    if (type === 'separate') {
        try {
            await performSeparation(data);
        } catch (error) {
            self.postMessage({
                type: 'error',
                data: { message: error.message }
            });
        }
    }
};

/**
 * Perform ICA separation
 * @param {Object} data - Contains mixedSignals, algorithm, settings
 */
async function performSeparation(data) {
    const { mixedSignals, algorithm, settings } = data;
    
    // Convert to Float32Array[]
    const signals = mixedSignals.map(s => new Float32Array(s));
    
    // Progress callback
    const progressCallback = (progress) => {
        self.postMessage({
            type: 'progress',
            data: progress
        });
    };
    
    // Step 1: Preprocessing
    progressCallback({
        step: 'preprocessing',
        progress: 0.1,
        message: 'Preprocessing data...'
    });
    
    const preprocessor = new Preprocessor();
    const whitened = preprocessor.preprocess(signals);
    
    // Step 2: ICA
    let result;
    
    if (algorithm === 'fastica') {
        const fastica = new FastICA({
            maxIterations: settings.icaSettings.maxIterations,
            tolerance: settings.icaSettings.tolerance
        });
        
        result = fastica.fit(whitened, (icaProgress) => {
            progressCallback({
                step: 'ica',
                progress: 0.2 + 0.7 * (icaProgress.iteration / settings.icaSettings.maxIterations),
                message: `FastICA iteration ${icaProgress.iteration}`,
                ...icaProgress
            });
        });
    } else if (algorithm === 'jade') {
        const jade = new JADE({
            maxIterations: settings.icaSettings.maxIterations,
            tolerance: settings.icaSettings.tolerance
        });
        
        result = jade.fit(whitened, (jadeProgress) => {
            progressCallback({
                step: jadeProgress.step || 'ica',
                progress: jadeProgress.progress || 0.5,
                message: `JADE ${jadeProgress.step || 'processing'}...`,
                ...jadeProgress
            });
        });
    }
    
    // Convert sources to transferable arrays
    const sources = result.sources.map(s => Array.from(s));
    
    // Send results back
    self.postMessage({
        type: 'complete',
        data: {
            sources: sources,
            unmixingMatrix: result.unmixingMatrix,
            algorithm: algorithm,
            iterations: result.iterations,
            converged: result.converged
        }
    });
}