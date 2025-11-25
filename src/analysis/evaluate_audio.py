import argparse
import numpy as np
import matplotlib.pyplot as plt
import wave
import os

def evaluate_audio(wav_file, output_dir):
    """
    Evaluates audio quality and generates a spectrogram.
    
    Args:
        wav_file (str): Path to the .wav file.
        output_dir (str): Directory to save analysis results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with wave.open(wav_file, 'rb') as wf:
            params = wf.getparams()
            nchannels, sampwidth, framerate, nframes, comptype, compname = params
            
            print(f"Audio File: {wav_file}")
            print(f"Channels: {nchannels}, Rate: {framerate}, Frames: {nframes}")
            
            str_data = wf.readframes(nframes)
            wave_data = np.frombuffer(str_data, dtype=np.int16)
            
            # Reshape if multiple channels
            if nchannels > 1:
                wave_data = wave_data.reshape(-1, nchannels)
                # Analyze first channel for simplicity
                channel_data = wave_data[:, 0]
            else:
                channel_data = wave_data
                
            # 1. Calculate RMS (Volume Level)
            rms = np.sqrt(np.mean(channel_data**2))
            print(f"RMS Level: {rms:.2f}")
            
            if rms < 100:
                print("WARNING: Audio level is very low. Check microphone gain.")
            
            # 2. Generate Spectrogram
            plt.figure(figsize=(10, 4))
            plt.specgram(channel_data, Fs=framerate, NFFT=1024, noverlap=512, cmap='inferno')
            plt.title(f"Spectrogram: {os.path.basename(wav_file)}")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.colorbar(format='%+2.0f dB')
            
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(wav_file))[0]}_spectrogram.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved spectrogram to {output_path}")
            
    except Exception as e:
        print(f"Error analyzing audio: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Audio Quality")
    parser.add_argument("wav_file", help="Path to input .wav file")
    parser.add_argument("--output", default="analysis_results", help="Output directory")
    args = parser.parse_args()
    
    evaluate_audio(args.wav_file, args.output)
