import numpy as np
import matplotlib.pyplot as plt
import time
import os

save_folder = "task2_plots"
os.makedirs(save_folder, exist_ok=True)

def generate_random_signal(n):
    """Generate a random discrete signal of length n."""
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

# def fft(signal):
#     N = len(signal)
#     if N <= 1:
#         return signal
#     even = fft(signal[0::2])
#     odd = fft(signal[1::2])
#     T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
#     return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# def ifft(spectrum):
#     N = len(spectrum)
#     if N <= 1:
#         return spectrum
#     even = ifft(spectrum[0::2])
#     odd = ifft(spectrum[1::2])
#     T = [np.exp(2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
#     return [(even[k] + T[k]) / 2 for k in range(N // 2)] + [(even[k] - T[k]) / 2 for k in range(N // 2)]

def fft(x):
    """
    Recursive implementation of the Fast Fourier Transform (FFT).
    Input:  Array of complex numbers x (length N).
    Output: Array of complex numbers representing the FFT of x.
    """
    N = len(x)
    
    # Step 1: Zero Padding (if necessary)
    if N & (N - 1) != 0:  # Check if N is not a power of 2
        next_power_of_2 = 1 << (N - 1).bit_length()  # Find the next power of 2
        x = np.pad(x, (0, next_power_of_2 - N))  # Pad with zeros
        N = next_power_of_2

    # Step 2: Base Case
    if N == 1:
        return x
    
    # Step 3: Divide the input array into even and odd indexed parts
    even = fft(x[::2])  # FFT of even indices
    odd = fft(x[1::2])  # FFT of odd indices
    
    # Step 5: Prepare the result array
    result = np.zeros(N, dtype=complex)
    
    # Step 6: Calculate the twiddle factors and combine the results
    for k in range(N // 2):
        twiddle_factor = np.exp(-2j * np.pi * k / N)
        result[k] = even[k] + twiddle_factor * odd[k]
        result[k + N // 2] = even[k] - twiddle_factor * odd[k]
    
    # Step 7: Return the combined result
    return result

def ifft(x):
    """
    Recursive implementation of the Inverse Fast Fourier Transform (IFFT).
    Input:  Array of complex numbers x (length N).
    Output: Array of complex numbers representing the IFFT of x.
    """
    N = len(x)

    # Step 1: Zero Padding (if necessary)
    if N & (N - 1) != 0:  # Check if N is not a power of 2
        next_power_of_2 = 1 << (N - 1).bit_length()  # Find the next power of 2
        x = np.pad(x, (0, next_power_of_2 - N))  # Pad with zeros
        N = next_power_of_2

    # Step 2: Base Case
    if N == 1:
        return x

    # Step 3: Divide the input array into even and odd indexed parts
    even = ifft(x[::2])  # IFFT of even indices
    odd = ifft(x[1::2])  # IFFT of odd indices

    # Step 5: Prepare the result array
    result = np.zeros(N, dtype=complex)

    # Step 6: Calculate the twiddle factors and combine the results
    for k in range(N // 2):
        twiddle_factor = np.exp(2j * np.pi * k / N)  # Conjugate of FFT twiddle factor
        result[k] = even[k] + twiddle_factor * odd[k]
        result[k + N // 2] = even[k] - twiddle_factor * odd[k]

    # Step 7: Normalize by dividing by N
    return result / 2



# Measure and Compare Runtime
def measure_runtime(n_values):
    dft_times = []
    fft_times = []
    idft_times = []
    ifft_times = []

    total_times = 1

    for n in n_values:
        signal = generate_random_signal(n)
        
        # Measure DFT Runtime
        dft_time = 0
        for _ in range(total_times):
            start = time.perf_counter()
            dft_result = dft(signal)
            dft_time += (time.perf_counter() - start)
        dft_times.append(dft_time / total_times)
        
        # Measure IDFT Runtime
        idft_time = 0
        for _ in range(total_times):
            start = time.perf_counter()
            idft_result = idft(dft_result)
            idft_time += (time.perf_counter() - start)
        idft_times.append(idft_time / total_times)

        # Measure FFT Runtime
        fft_time = 0
        for _ in range(total_times):
            start = time.perf_counter()
            fft_result = fft(signal)
            fft_time += (time.perf_counter() - start)
        fft_times.append(fft_time / total_times)
        
        # Measure IFFT Runtime
        ifft_time = 0
        for _ in range(total_times):
            start = time.perf_counter()
            ifft_result = ifft(fft_result)
            ifft_time += (time.perf_counter() - start)
        ifft_times.append(ifft_time / total_times)

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
    # plt.show()
    plt.savefig(f"{save_folder}/{label1}_vs_{label2}.png")
    plt.close()

# Plot the results
n_values = [k for k in range(1, 1000)]  # n = 4, 8, 16, ..., 1024
dft_times, fft_times, idft_times, ifft_times = measure_runtime(n_values)
print(dft_times)
print(fft_times)
plot_dft_fft(n_values, dft_times, fft_times)
plot_dft_fft(n_values, idft_times, ifft_times,label1="IDFT",label2="IFFT")

