"""Worker script for audio processing"""
from js import console
from pyscript import sync

console.log("Worker: Initializing...")

# Import SJPlot module
console.log("Worker: Importing SJPlot...")
import SJPlot
console.log("Worker: SJPlot imported successfully")

def process_audio():
    """Process audio in worker thread"""
    try:
        console.log("Worker: Starting audio processing...")
        SJPlot.main()
        console.log("Worker: Processing complete")
    except Exception as e:
        console.error(f"Worker error: {e}")
        import traceback
        console.error(traceback.format_exc())
        raise

# Export function to main thread
__export__ = ['process_audio']
console.log("Worker: Ready and exported process_audio")
