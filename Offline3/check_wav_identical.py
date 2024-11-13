from scipy.io import wavfile
import numpy as np

def compare_wav_files_percentage(file1, file2):
    # Read the two .wav files
    sample_rate1, data1 = wavfile.read(file1)
    sample_rate2, data2 = wavfile.read(file2)
    
    # Check if the sample rates are the same
    if sample_rate1 != sample_rate2:
        print("The sample rates are different.")
        return
    
    # Check if the data dimensions match
    if data1.shape != data2.shape:
        print("The audio data dimensions are different.")
        return
    
    # Calculate the number of identical samples
    identical_samples = np.sum(data1 == data2)
    
    # Calculate the total number of samples
    total_samples = data1.size
    
    # Calculate the percentage of identical samples
    percentage_identical = (identical_samples / total_samples) * 100
    
    print(f"The percentage of identical samples is: {percentage_identical:.2f}%")

# Usage example
file1 = 'denoised_audio.wav'
file2 = 'denoised_audio_main.wav'

compare_wav_files_percentage(file1, file2)
