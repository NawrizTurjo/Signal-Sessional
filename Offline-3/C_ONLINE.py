import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

# Step 1: Define the given function f(t)
def f(t):
    return 2 * np.sin(14 * np.pi * t) - np.sin(2 * np.pi * t) * (4 * np.sin(2 * np.pi * t) * np.sin(14 * np.pi * t) - 1)

# Time vector
t = np.linspace(0, 1, 1000, endpoint=False)  # 1 second sampled at 1000 Hz
f_t = f(t)

# Step 1.1: Plot the original function
plt.figure(figsize=(12, 6))
plt.plot(t, f_t, label='Original Function f(t)')
plt.title("Original Function f(t)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Step 2: Compute the Fourier Transform
N = len(t)  # Number of samples
dt = t[1] - t[0]  # Sampling interval
frequencies = fftfreq(N, dt)  # Frequency values
fft_result = fft(f_t)  # Compute FFT

# Magnitude spectrum
magnitude = np.abs(fft_result)

# Step 2.1: Plot the frequency spectrum
plt.figure(figsize=(12, 6))
plt.plot(frequencies[:N // 2], magnitude[:N // 2], label="Magnitude Spectrum")
plt.title("Frequency Spectrum of f(t)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()
plt.show()

# Step 3: Identify significant frequencies
threshold = 0.1 * np.max(magnitude)  # Define a threshold for peak detection
significant_freqs = frequencies[np.where(magnitude > threshold)]

# Print significant frequencies
print("Significant Frequencies (Hz):", significant_freqs[significant_freqs > 0])  # Ignore negative frequencies

# Step 4: Reconstruct f(t) using the identified frequencies
reconstructed_f_t = np.zeros_like(f_t)
for freq in significant_freqs[significant_freqs > 0]:
    reconstructed_f_t += np.cos(2 * np.pi * freq * t)  # Use cosine since phase is 0

# Step 4.1: Plot the reconstructed function
plt.figure(figsize=(12, 6))
plt.plot(t, reconstructed_f_t, label='Reconstructed Function')
plt.plot(t, f_t, linestyle='dashed', label='Original Function', alpha=0.7)
plt.title("Comparison of Original and Reconstructed Functions")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Step 5: Verify if the reconstructed function matches the original
error = np.abs(f_t - reconstructed_f_t).max()
print("Maximum reconstruction error:", error)
