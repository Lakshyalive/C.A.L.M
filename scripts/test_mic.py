import sounddevice as sd
import numpy as np

print("Testing microphone for 5 seconds...")
print("Speak now and watch the levels:\n")

def callback(indata, frames, time, status):
    rms = np.sqrt(np.mean(indata**2))
    bars = int(rms * 1000)
    print(f"Level: {'█' * bars:<30} {rms:.4f}", end='\r')

with sd.InputStream(
    samplerate = 16000,
    channels   = 1,
    dtype      = 'float32',
    callback   = callback
):
    sd.sleep(5000)

print("\n\nDone. If you saw bars moving when you spoke, mic is working.")
print("Note the RMS value when speaking — you'll need it for threshold setting.")
