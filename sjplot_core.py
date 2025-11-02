"""
SJPlot Online - Browser Version
Simplified core for PyScript
"""

import numpy as np
from scipy.signal import find_peaks, sosfiltfilt, iirfilter, lfilter
from scipy.ndimage import uniform_filter1d
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
import io
import base64
from js import console

__version__ = "Online-1.0"

def process_audio(file0_data, file1_data, config):
    """
    Main processing function called from JavaScript
    """
    try:
        console.log("Starting audio processing...")
        
        # Parse audio files
        audio0, fs0 = parse_wav_bytes(file0_data['data'])
        console.log(f"File 0 loaded: {fs0}Hz, {len(audio0)} samples")
        
        audio1 = None
        if file1_data:
            audio1, fs1 = parse_wav_bytes(file1_data['data'])
            console.log(f"File 1 loaded: {fs1}Hz, {len(audio1)} samples")
        
        # Process and generate plot
        plot_html = generate_plot(audio0, audio1, fs0, config)
        
        return plot_html
        
    except Exception as e:
        console.error(f"Error in process_audio: {str(e)}")
        raise


def parse_wav_bytes(byte_data):
    """Parse WAV file from bytes"""
    # Convert JS Uint8Array to bytes
    wav_bytes = bytes(byte_data)
    
    # Use scipy to read WAV
    with io.BytesIO(wav_bytes) as wav_io:
        fs, audio = wavfile.read(wav_io)
    
    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    
    return audio, fs


def ft_window(n):
    """Flat-top window (Matlab style)"""
    a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
    x = np.arange(n)
    pi = np.pi
    n_minus_1 = n - 1
    
    w = (a0 - a1 * np.cos(2*pi*x/n_minus_1) + 
         a2 * np.cos(4*pi*x/n_minus_1) - 
         a3 * np.cos(6*pi*x/n_minus_1) + 
         a4 * np.cos(8*pi*x/n_minus_1))
    return w


def rfft_analysis(signal, fs, minf, maxf, fstep):
    """Simplified FFT analysis"""
    F = int(fs / fstep)
    win = ft_window(F)
    
    if len(signal.shape) == 1:
        signal = np.expand_dims(signal, axis=0)
    
    freq, amp = [], []
    
    for x in range(0, signal.shape[1] - F, F):
        y0 = np.abs(np.fft.rfft(signal[0, x:x + F] * win))
        f0 = np.argmax(y0)
        
        if minf/fstep <= f0 <= maxf/fstep:
            freq.append(f0 * fstep)
            amp.append(y0[f0])
    
    return freq, amp


def create_plot_data(signal, fs, normalize_freq=1000):
    """Create frequency response data from signal"""
    # Process in chunks
    freq_all, amp_all = [], []
    
    # Low frequency chunk
    f, a = rfft_analysis(signal, fs, 20, 90, 10)
    freq_all.extend(f)
    amp_all.extend([20 * np.log10(amp) - 19.995 for amp in a])
    
    # Mid frequency chunk  
    f, a = rfft_analysis(signal, fs, 100, 980, 20)
    freq_all.extend(f)
    amp_all.extend([20 * np.log10(amp) - 13.99 for amp in a])
    
    # High frequency chunk
    f, a = rfft_analysis(signal, fs, 1000, 20000, 100)
    freq_all.extend(f)
    amp_all.extend([20 * np.log10(amp) for amp in a])
    
    # Normalize
    norm_idx = np.argmin(np.abs(np.array(freq_all) - normalize_freq))
    norm_val = amp_all[norm_idx]
    amp_all = [a - norm_val for a in amp_all]
    
    # Apply smoothing
    sos = iirfilter(3, 0.5, btype='lowpass', output='sos')
    amp_all = sosfiltfilt(sos, amp_all)
    
    return freq_all, amp_all


def generate_plot(audio0, audio1, fs, config):
    """Generate matplotlib plot and return as HTML"""
    console.log("Generating plot...")
    
    # Ensure audio is transposed correctly
    if len(audio0.shape) == 1:
        audio0 = audio0.reshape(1, -1)
    else:
        audio0 = audio0.T
    
    # Create plot data
    freq0, amp0 = create_plot_data(audio0, fs, config['normalize'])
    
    freq1, amp1 = None, None
    if audio1 is not None:
        if len(audio1.shape) == 1:
            audio1 = audio1.reshape(1, -1)
        else:
            audio1 = audio1.T
        freq1, amp1 = create_plot_data(audio1, fs, config['normalize'])
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    if config['plot_style'] == 4:
        # Dual plot with zoom
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        
        # Top plot - zoomed
        ax1.semilogx(freq0, amp0, color='#0000ff', label='Left/File 0')
        if freq1 is not None:
            ax1.semilogx(freq1, amp1, color='#ff0000', label='Right/File 1')
        
        ax1.set_ylabel("Amplitude (dB)")
        ax1.set_ylim(-5, 5)
        ax1.legend(loc=4)
        ax1.grid(True, which="major", ls="-", color="black", alpha=0.3)
        ax1.grid(True, which="minor", ls="-", color="gainsboro", alpha=0.2)
        
        # Bottom plot - full range  
        ax2.semilogx(freq0, amp0, color='#0000ff', label='Left/File 0')
        if freq1 is not None:
            ax2.semilogx(freq1, amp1, color='#ff0000', label='Right/File 1')
        
        ax2.set_ylabel("Amplitude (dB)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.legend(loc=4)
        ax2.grid(True, which="major", ls="-", color="black", alpha=0.3)
        ax2.grid(True, which="minor", ls="-", color="gainsboro", alpha=0.2)
        
        # Set x-axis ticks
        for ax in [ax1, ax2]:
            ax.set_xticks([20, 50, 100, 500, 1000, 5000, 10000, 20000])
            ax.set_xticklabels(['20', '50', '100', '500', '1k', '5k', '10k', '20k'])
    
    else:
        # Single plot (style 1 or 5)
        ax = plt.subplot(111)
        ax.semilogx(freq0, amp0, color='#0000ff', linewidth=2, label='Left/File 0')
        
        if freq1 is not None:
            ax.semilogx(freq1, amp1, color='#ff0000', linewidth=2, label='Right/File 1')
        
        ax.set_ylabel("Amplitude (dB)", fontsize=12)
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylim(-5, 5)
        ax.legend(loc=4, fontsize=10)
        ax.grid(True, which="major", ls="-", color="black", alpha=0.3)
        ax.grid(True, which="minor", ls="-", color="gainsboro", alpha=0.2)
        ax.set_xticks([20, 50, 100, 500, 1000, 5000, 10000, 20000])
        ax.set_xticklabels(['20', '50', '100', '500', '1k', '5k', '10k', '20k'])
    
    plt.title("SJPlot Online - Frequency Response Analysis", fontsize=16, pad=20)
    plt.tight_layout()
    
    # Convert to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    
    # Return as HTML
    html = f'''
    <div class="plot-container">
        <h3>Analysis Results</h3>
        <img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 100%;" />
        <div style="margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 8px;">
            <p><strong>Configuration:</strong></p>
            <p>Plot Style: {config['plot_style']} | RIAA Mode: {config['riaa_mode']} | Normalize: {config['normalize']}Hz</p>
        </div>
    </div>
    '''
    
    return html
