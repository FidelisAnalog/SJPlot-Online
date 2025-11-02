# SJPlot Online - AI Coding Assistant Instructions

## Project Overview
SJPlot Online is a browser-based phono cartridge frequency response analyzer built with PyScript. It processes WAV audio files through advanced FFT analysis to generate frequency response plots for audio equipment testing.

## Architecture & Core Components

### Hybrid JavaScript-Python Structure
- **`index.html`**: Single-page application containing all UI, styling, PyScript configuration, and JavaScript-Python integration
- **`sjplot_core.py`**: Simplified Python core (currently unused but maintained for reference)
- **PyScript Integration**: Embedded Python runs in browser via `<py-script>` tags with scientific computing packages

### Key Technical Patterns

#### PyScript Configuration & Dependencies
```html
<py-config>
    packages = ["numpy", "scipy", "matplotlib"]
</py-config>
```
Always specify exact package dependencies in `<py-config>`. PyScript loads these from PyPI at runtime.

#### JavaScript ↔ Python Data Bridge
Critical pattern for file handling:
```javascript
// JavaScript side - convert file to bytes
const data = new Uint8Array(e.target.result);
const pyFileData = window.pyscript.interpreter.globals.get('file_data');
pyFileData.set('file0', data);

// Python side - access JS data
wav0_bytes = bytes(file_data['file0'])
with io.BytesIO(wav0_bytes) as wav_io:
    Fs, audio0 = wavfile.read(wav_io)
```

#### Audio Processing Pipeline (Embedded in HTML)
1. **WAV File Parsing**: `scipy.io.wavfile` with `io.BytesIO` wrapper
2. **Frequency Analysis**: Custom FFT implementation using `ft_window()` (5-term Blackman-Nuttall)
3. **Multi-Band Processing**: Hardcoded frequency ranges `[(20,45,5), (50,90,10), (100,980,20), (1000,20000,100)]`
4. **RIAA Filtering**: IIR coefficients for 96kHz only: `riaaiir()` function
5. **Plot Generation**: `matplotlib` → base64 → HTML injection

#### Plot Styles & Visualization
- **Style 4 (default)**: Dual subplot with harmonics analysis and crosstalk detection
- **Styles 1/5**: Single frequency response plot
- **Color Scheme**: Blue (#0000ff) for left/primary, Red (#ff0000) for right/secondary
- **Harmonics**: 2nd harmonic (#0080ff), 3rd harmonic (#00dfff)

## Critical Development Patterns

### Audio Data Handling
Audio must be transposed after loading: `audio0 = audio0.T`
Support int16/int32 → float32 conversion with proper scaling factors.

### Error Handling Strategy
```python
try:
    # processing
except Exception as e:
    import traceback
    console.error(traceback.format_exc())
    # Update DOM with error state
```
Always use `console.log()` and `console.error()` for PyScript debugging.

### DOM Manipulation from Python
```python
from js import document
document.getElementById('status').textContent = 'Processing...'
document.getElementById('loading').classList.add('active')
```

### Configuration Management
Configuration object passed from JS to Python:
```javascript
const config = {
    plot_style: parseInt(document.getElementById('plotStyle').value),
    riaa_mode: parseInt(document.getElementById('riaaMode').value),
    riaa_inverse: document.getElementById('riaaInverse').checked ? 1 : 0,
    normalize: parseInt(document.getElementById('normalize').value)
};
```

## Development Workflow

### Local Testing
- Open `index.html` directly in browser (no server required)
- PyScript loads dependencies automatically from CDN
- Use browser DevTools Console for PyScript debugging
- Audio files must be local WAV files (security restrictions)

### Debugging PyScript Issues
1. Check browser console for PyScript loading errors
2. Verify package imports in `<py-config>`
3. Use `console.log()` extensively in Python code
4. Test JavaScript-Python data transfer with simple values first

### Performance Considerations
- FFT processing is CPU-intensive in browser
- Large audio files may cause browser freezing
- Consider chunked processing for files >10MB
- PyScript startup time ~3-5 seconds for package loading

## File Modification Guidelines

### Adding New Analysis Features
1. Extend `process_audio()` function in `<py-script>` section
2. Update configuration object in JavaScript `startAnalysis()`
3. Modify plot generation in matplotlib section
4. Update UI elements in HTML accordingly

### UI/Styling Changes
All styling is embedded in `<style>` section of `index.html`. Uses CSS Grid for responsive configuration layout.

### Scientific Computing
Leverage existing FFT pipeline in `rfft_full()` and `createplotdata()` functions. These implement domain-specific audio analysis patterns optimized for phono cartridge testing.

## External Dependencies
- **PyScript 2024.1.1**: Core Python-in-browser runtime
- **SciPy**: Signal processing and file I/O
- **NumPy**: Numerical computations
- **Matplotlib**: Plot generation and base64 conversion
- **Original SJPlot**: Reference implementation at github.com/FidelisAnalog/SJPlot

## Testing Audio Files
Use 96kHz WAV files for optimal RIAA filter performance. Lower sample rates may produce suboptimal results due to hardcoded filter coefficients.