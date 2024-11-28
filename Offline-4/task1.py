import numpy as np
import matplotlib.pyplot as plt
import os

save_folder = "task1_plots"
os.makedirs(save_folder, exist_ok=True)
saveToFolder = False

n=50
samples = np.arange(n) 
sampling_rate=100
wave_velocity=8000



#use this function to generate signal_A and signal_B with a random shift
def generate_signals(frequency=5, noise_freqs=[15, 30, 45], amplitudes=[0.5, 0.3, 0.1], 
                     noise_freqs2=[10, 20, 40], amplitudes2=[0.3, 0.2, 0.1]):

    # noise_freqs = [15, 30, 45]  # Default noise frequencies in Hz

    # amplitudes = [0.5, 0.3, 0.1]  # Default noise amplitudes
    # noise_freqs2 = [10, 20, 40] 
    # amplitudes2 = [0.3, 0.2, 0.1]
    
     # Discrete sample indices
    dt = 1 / sampling_rate  # Sampling interval in seconds
    time = samples * dt  # Time points corresponding to each sample

    # Original clean signal (sinusoidal)
    original_signal = np.sin(2 * np.pi * frequency * time)

    # Adding noise
    noise_for_sigal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_sigal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))
    signal_A = original_signal + noise_for_sigal_A 
    noisy_signal_B = signal_A + noise_for_sigal_B

    # Applying random shift
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    shift_samples = 3
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples)
    
    return signal_A, signal_B


def dft(signal):
    N = len(signal)
    X = []
    for k in range(N):
        real = sum(signal[n] * np.cos(2 * np.pi * k * n / N) for n in range(N))
        imag = -sum(signal[n] * np.sin(2 * np.pi * k * n / N) for n in range(N))
        X.append(complex(real, imag))
    return X

def idft(X):
    N = len(X)
    signal = []
    for n in range(N):
        real = sum(X[k].real * np.cos(2 * np.pi * k * n / N) - X[k].imag * np.sin(2 * np.pi * k * n / N) for k in range(N))
        signal.append(real / N)
    return signal

def cross_correlation_dft(signal_A, signal_B):
    dft_A = dft(signal_A)
    dft_B = dft(signal_B)
    # conjugate_dft_B = [np.conj(b) for b in dft_B]
    
    # cross_corr_freq = [a * b for a, b in zip(dft_A, conjugate_dft_B)]
    cross_corr_freq = dft_A * np.conj(dft_B)
    cross_corr_freq = dft_B * np.conj(dft_A)
    
    cross_corr_time = idft(cross_corr_freq)
    cross_corr_time = np.roll(cross_corr_time, len(cross_corr_time) // 2)
    return cross_corr_time

def find_sample_lag(cross_corr):
    lag_index = np.argmax(np.real(cross_corr))  # Use real part only
    if lag_index > len(cross_corr) // 2:
        lag_index -= len(cross_corr)
    return lag_index

def estimate_distance(sample_lag, sampling_rate, wave_velocity):
    time_delay = abs(sample_lag) / sampling_rate
    distance = time_delay * wave_velocity
    return distance

def plot_single_signal(signal, title, color):
    plt.figure(figsize=(12, 4))
    plt.stem(signal, linefmt=f"{color}-", markerfmt=f"{color}o", basefmt=" ")
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    if saveToFolder:
        plt.savefig(f"{save_folder}/{title}.png")
        plt.close()
    else:
        plt.show()

def plot_cross_correlation_dft(cross_corr,label="Cross Correlation"):
    lags = np.arange(-n//2, n//2)
    plt.figure(figsize=(12, 4))
    plt.stem(lags, cross_corr, linefmt="g-", markerfmt="go", basefmt=" ")
    plt.title(f"{label}")
    plt.xlabel("Lag (Samples)")
    plt.ylabel("Correlation")
    if saveToFolder:
        plt.savefig(f"{save_folder}/{label}.png")
        plt.close()
    else:
        plt.show()

from scipy.signal import butter, lfilter

# Low-pass filter
def low_pass_filter(signal, cutoff=10, sampling_rate=100, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


def main():
    # Generate signals
    signal_A, signal_B = generate_signals()
    dft_signal_A = np.abs(dft(signal_A))
    dft_signal_B = np.abs(dft(signal_B))

    # Cross-correlation
    cross_corr = cross_correlation_dft(signal_A, signal_B)

    # Find sample lag
    sample_lag = find_sample_lag(cross_corr)

    # Estimate distance
    distance = estimate_distance(sample_lag, sampling_rate, wave_velocity)

    # Print results
    print(f"Sample Lag: {sample_lag}")
    print(f"Estimated Distance: {distance:.2f} meters")

    # Plot results
    # plot_signals(signal_A, signal_B, cross_corr)
    plot_single_signal(signal_A, "Signal A","b")
    plot_single_signal(signal_B, "Signal B","r")
    plot_single_signal(dft_signal_A, "DFT of Signal A","b")
    plot_single_signal(dft_signal_B, "DFT of Signal B","r")
    plot_cross_correlation_dft(cross_corr)

    # New Signals for Testing
    signal_A, noisy_signal_B = generate_signals(
        noise_freqs=[15, 35, 50],
        amplitudes=[0.6, 0.4, 0.2],
        noise_freqs2=[12, 25, 45],
        amplitudes2=[0.5, 0.3, 0.2]
    )

    # Apply low-pass filtering
    filtered_signal_B = low_pass_filter(noisy_signal_B)
    plot_single_signal(signal_A, "Signal A","b")
    plot_single_signal(noisy_signal_B, "Noisy Signal B","r")
    plot_single_signal(filtered_signal_B, "Filtered Signal B","g")

    # Cross-correlation with noisy and filtered signals
    cross_corr_noisy = cross_correlation_dft(signal_A, signal_B)
    cross_corr_filtered = cross_correlation_dft(signal_A, filtered_signal_B)

    # Plot cross-correlations
    plot_cross_correlation_dft(cross_corr_noisy, label="Cross Correlation Noisy")
    plot_cross_correlation_dft(cross_corr_filtered, label="Cross Correlation Filtered")


if __name__ == "__main__":
    main()
