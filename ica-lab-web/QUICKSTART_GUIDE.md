# Quick Start Guide

## Running the Application

### Option 1: Local Server (Recommended)

The application requires a local web server due to ES6 module imports.

**Using Python 3:**
```bash
cd ica-lab
python -m http.server 8000
```

**Using Node.js:**
```bash
cd ica-lab
npx serve
```

**Using PHP:**
```bash
cd ica-lab
php -S localhost:8000
```

Then open your browser to `http://localhost:8000`

### Option 2: Live Server (VS Code)

1. Install "Live Server" extension in VS Code
2. Right-click on `index.html`
3. Select "Open with Live Server"

### Option 3: Deploy Online

**Netlify (Easiest):**
1. Go to netlify.com
2. Drag and drop the `ica-lab` folder
3. Get instant live URL

**Vercel:**
```bash
cd ica-lab
npx vercel
```

**GitHub Pages:**
1. Create GitHub repository
2. Push the `ica-lab` folder
3. Enable GitHub Pages in settings

## First Steps

### 1. Select Sources (at least 2)
- Click on preset sound cards to select them
- OR upload your own audio files
- Only non-Gaussian sources (green/yellow indicator) can be selected
- Red indicator = too Gaussian, cannot be used for ICA

### 2. Choose Algorithm
- **FastICA**: Fast, works with any number of sources
- **JADE**: More accurate, best for ≤6 sources

### 3. Mix Signals
Click "Mix Signals" button:
- Generates random mixing matrix
- Creates mixed observations
- You can play the mixed signals

### 4. Separate Sources
Click "Separate Sources" button:
- Runs ICA algorithm
- Shows progress bar
- Automatically matches recovered sources to originals

### 5. Compare Results
- Play original sources
- Play mixed signals  
- Play separated sources
- Check quality metrics

### 6. Reset
Click "Reset System" to start over

## Understanding the Indicators

### Kurtosis Values (κ)
- **κ < 0.1**: Nearly Gaussian (❌ cannot use)
- **κ 0.1-0.5**: Weakly non-Gaussian (⚠️ may work)
- **κ > 0.5**: Strongly non-Gaussian (✅ ideal)

### Separation Quality
- **Excellent** (90%+): Nearly perfect separation
- **Good** (70-90%): Good separation quality
- **Fair** (50-70%): Acceptable separation
- **Poor** (<50%): Poor separation quality

## Tips for Best Results

1. **Use Non-Gaussian Sources**
   - Music, speech, instruments work well
   - Pure sine waves may work but not ideal
   - White noise won't work (it's Gaussian)

2. **Choose Right Algorithm**
   - 2-6 sources → JADE for best accuracy
   - >6 sources → FastICA for speed

3. **Audio Quality**
   - Use clean audio without compression artifacts
   - 16kHz sample rate is optimal (automatic)
   - 5-20 seconds duration recommended

4. **Mixing**
   - More sources = more challenging separation
   - Start with 2-3 sources to learn
   - Gradually increase complexity

## Example Workflow

**Separating Music Tracks:**

1. Upload 3 audio files:
   - vocals.mp3
   - drums.mp3
   - bass.mp3

2. Select all three (check they're non-Gaussian)

3. Choose FastICA algorithm

4. Click "Mix Signals"
   - Listen to the 3 mixed signals
   - Note: Original tracks are now mixed together

5. Click "Separate Sources"
   - Wait 5-10 seconds for processing
   - View the separated tracks

6. Compare:
   - Original vocals vs. Separated vocals
   - Check confidence percentages
   - Evaluate separation quality

## Common Issues

### "Nearly Gaussian" Error
- **Cause**: Source has kurtosis < 0.1
- **Solution**: Use a different source or add distortion

### JADE is Slow
- **Cause**: Too many sources for JADE
- **Solution**: Use FastICA or reduce to ≤6 sources

### Poor Separation Quality
- **Possible causes**:
  - Sources are too similar
  - Sources are weakly non-Gaussian
  - Insufficient signal length
- **Solutions**:
  - Use more distinct sources
  - Increase audio duration
  - Try different algorithm

### Page Doesn't Load
- **Cause**: ES6 modules require server
- **Solution**: Use local server, don't open file:// directly

## Keyboard Shortcuts

- **Spacebar**: Play/pause current audio
- **Escape**: Stop all audio playback
- **Ctrl/Cmd + R**: Reset system (with confirmation)

## Performance Tips

**For Large Files:**
- Use JADE only with ≤6 sources
- FastICA handles 10+ sources efficiently
- Consider 10-15 second clips for faster processing

**For Best Accuracy:**
- Use JADE with 2-4 sources
- Ensure high kurtosis values (>0.5)
- Use uncompressed or lossless audio

## Browser Requirements

**Minimum:**
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

**Required Features:**
- Web Audio API
- ES6 Modules
- Canvas API
- Float32Array

## Next Steps

1. ✅ Run the demo with preset sounds
2. ✅ Upload your own audio
3. ✅ Compare FastICA vs JADE
4. ✅ Experiment with different numbers of sources
5. ✅ Read the full README.md for technical details

## Getting Help

- Check console (F12) for detailed error messages
- Review README.md for technical documentation
- Inspect network tab if files fail to load
- Ensure browser meets requirements

## Advanced Usage

**Custom Configuration:**
Edit `manifest.json` to:
- Change sample rate
- Adjust validation thresholds
- Modify iteration limits
- Add preset sounds

**Adding Sounds:**
1. Place audio files in `assets/sounds/`
2. Add entries to `manifest.json`
3. Reload application

**Debugging:**
Open browser console (F12) to see:
- Matrix computations
- Convergence details
- Performance metrics
- Error traces

---

**Ready to start?** Open the application and select your first sources!