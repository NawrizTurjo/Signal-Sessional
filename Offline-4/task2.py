import numpy as np
import matplotlib.pyplot as plt
import time
import os

save_folder = "task2_plots"
os.makedirs(save_folder, exist_ok=True)
saveToFolder = False

def generate_random_signal(n):
    return np.random.rand(n)

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

## Complex form implementation
# def dft(signal):
#     N = len(signal)
#     dft_output = np.zeros(N, dtype=complex)
#     for k in range(N):
#         for n in range(N):
#             dft_output[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
#     return dft_output

# def idft(signal_freq):
#     N = len(signal_freq)
#     idft_output = np.zeros(N, dtype=complex)
#     for n in range(N):
#         for k in range(N):
#             idft_output[n] += signal_freq[k] * np.exp(2j * np.pi * k * n / N)
#     return idft_output / N

def fft(x):

    N = len(x)
    
    if N & (N - 1) != 0:
        next_power_of_2 = 1 << (N - 1).bit_length()
        x = np.pad(x, (0, next_power_of_2 - N))
        N = next_power_of_2

    if N == 1:
        return x
    
    even = fft(x[::2])
    odd = fft(x[1::2])
    
    result = np.zeros(N, dtype=complex)
    
    for k in range(N // 2):
        twiddle_factor = np.exp(-2j * np.pi * k / N)
        result[k] = even[k] + twiddle_factor * odd[k]
        result[k + N // 2] = even[k] - twiddle_factor * odd[k]

    return result

def ifft(x):

    N = len(x)

    if N & (N - 1) != 0:
        next_power_of_2 = 1 << (N - 1).bit_length() 
        x = np.pad(x, (0, next_power_of_2 - N))
        N = next_power_of_2

    if N == 1:
        return x

    even = ifft(x[::2])
    odd = ifft(x[1::2])

    result = np.zeros(N, dtype=complex)

    for k in range(N // 2):
        twiddle_factor = np.exp(2j * np.pi * k / N) 
        result[k] = even[k] + twiddle_factor * odd[k]
        result[k + N // 2] = even[k] - twiddle_factor * odd[k]

    return result / 2


def measure_runtime(n_values):
    dft_times = []
    fft_times = []
    idft_times = []
    ifft_times = []

    total_times = 2

    for n in n_values:
        signal = generate_random_signal(n)
        
        dft_time = 0
        for _ in range(total_times):
            start = time.perf_counter()
            dft_result = dft(signal)
            dft_time += (time.perf_counter() - start)
        dft_times.append(dft_time / total_times)
        
        idft_time = 0
        for _ in range(total_times):
            start = time.perf_counter()
            idft_result = idft(dft_result)
            idft_time += (time.perf_counter() - start)
        idft_times.append(idft_time / total_times)

        fft_time = 0
        for _ in range(total_times):
            start = time.perf_counter()
            fft_result = fft(signal)
            fft_time += (time.perf_counter() - start)
        fft_times.append(fft_time / total_times)
        
        ifft_time = 0
        for _ in range(total_times):
            start = time.perf_counter()
            ifft_result = ifft(fft_result)
            ifft_time += (time.perf_counter() - start)
        ifft_times.append(ifft_time / total_times)
        # plot_reconstructed_signals_vs_main(n_values, signal, [idft_result, ifft_result], title="Reconstructed Signals vs Main Signal")

    return dft_times, fft_times, idft_times, ifft_times

def plot_dft_fft(n_values, dft_times, fft_times,label1="DFT",label2="FFT"):
    plt.figure(figsize=(12, 6))
    plt.plot(n_values, dft_times, label=label1, marker='o', linestyle='-', color='blue')
    plt.plot(n_values, fft_times, label=label2, marker='o', linestyle='--', color='red')
    plt.title(f"{label1} vs {label2} Runtime Comparison")
    plt.xlabel("Signal Length (n)")
    plt.ylabel("Runtime (s)")
    plt.xscale("log")
    plt.yscale("log")
    log_n_values = np.log2(n_values)
    plt.xticks(n_values, [f"$2^{{{int(log_n)}}}$" for log_n in log_n_values])
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    if saveToFolder:
        plt.savefig(f"{save_folder}/{label1}_vs_{label2}.png")
        plt.close()
    else:
        plt.show()

def plot_reconstructed_signals_vs_main(n_values, signal, reconstructed_signals, title):
    plt.figure(figsize=(12, 6))
    plt.plot(signal, label="Main Signal", marker='o', linestyle='-', color='blue')
    for i, (n, reconstructed_signal) in enumerate(zip(n_values, reconstructed_signals)):
        plt.plot(reconstructed_signal, label=f"Reconstructed Signal (n={n})", marker='o', linestyle='--', color=f'C{i % 10}')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    if saveToFolder:
        plt.savefig(f"{save_folder}/reconstructed_signals_vs_main.png")
        plt.close()
    else:
        plt.show()

# Plot the results
n_values = [2**k for k in range(2, 12)]  # n = 4, 8, 16, ..., 1024
dft_times, fft_times, idft_times, ifft_times = measure_runtime(n_values)
# print(dft_times)
# print(fft_times)
plot_dft_fft(n_values, dft_times, fft_times)
plot_dft_fft(n_values, idft_times, ifft_times,label1="IDFT",label2="IFFT")

