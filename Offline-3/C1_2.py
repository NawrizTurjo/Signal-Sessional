import numpy as np
import matplotlib.pyplot as plt

# Define the interval and function: Parabolic Function y = x^2 within the interval [-2, 2]
x_values = np.linspace(-10, 10, 2000)
parabolic_values = np.where(np.abs(x_values) <= 2, x_values**2, 0)

# Plot the original function
# plt.figure(figsize=(12, 4))
# plt.plot(x_values, y_values, label="Original y = x^2")
# plt.title("Original Function (y = x^2)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()


triangular_values = np.where(np.abs(x_values) <= 2, 1 - np.abs(x_values) / 2, 0)

# Sawtooth Function within [-2, 2] with a slope of 1
# sawtooth_values = np.where(np.abs(x_values) <= 2, x_values , 0)
# sawtooth_values = np.where((x_values >= -2) & (x_values <= 2), (x_values + 2) / 2, 0)
sawtooth_values = np.where(np.abs(x_values) <= 2, (x_values + 2) / 4, 0)
# Rectangular Function within [-2, 2] with height 1
rectangular_values = np.where(np.abs(x_values) <= 2, 1, 0)

term1 = 2 * np.sin(14 * np.pi * x_values)
term2 = np.sin(2 * np.pi * x_values) * (4 * np.sin(2 * np.pi * x_values) * np.sin(14 * np.pi * x_values) - 1)

online_function = term1 - term2

# Plot Triangular Function
# plt.figure(figsize=(12, 4))
# plt.plot(x_values, triangular_values, label="Triangular Function", color="blue")
# plt.title("Triangular Function (Interval [-2, 2])")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# Plot Sawtooth Function
# plt.figure(figsize=(12, 4))
# plt.plot(x_values, sawtooth_values, label="Sawtooth Function", color="green")
# plt.title("Sawtooth Function (Interval [-2, 2])")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# Plot Rectangular Function
# plt.figure(figsize=(12, 4))
# plt.plot(x_values, rectangular_values, label="Rectangular Function", color="red")
# plt.title("Rectangular Function (Interval [-2, 2])")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# Fourier Transform function using trapezoidal integration
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    for i, freq in enumerate(frequencies):
        ft_result_real[i] = np.trapz(signal * np.cos(2 * np.pi * freq * sampled_times), sampled_times)
        ft_result_imag[i] = -np.trapz(signal *np.sin(2 * np.pi * freq * sampled_times), sampled_times)
    
    return ft_result_real, ft_result_imag



# Inverse Fourier Transform to reconstruct the signal
def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    
    for i, t in enumerate(sampled_times):
        cosine_part=np.cos(2 * np.pi * frequencies * t)
        sine_part=np.sin(2 * np.pi * frequencies * t)
        real_part_sum = np.trapz(ft_signal[0] * cosine_part, frequencies)
        imaginiary_part_sum = np.trapz(ft_signal[1] * sine_part, frequencies)
        reconstructed_signal[i] = real_part_sum - imaginiary_part_sum
    return reconstructed_signal


# Define frequency ranges and calculate FT, IFT, and plot for each
frequency_ranges = [1, 2, 5]
functions = [
    ("online Function",online_function,"gray")
    # ("Parabolic Function", parabolic_values, "black"),
    # ("Triangular Function", triangular_values, "blue"),
    # ("Sawtooth Function", sawtooth_values, "green"),
    # ("Rectangular Function", rectangular_values, "red")
]

for func_name, func_values, color in functions:
    # Plot the original function before processing
    plt.figure(figsize=(12, 4))
    plt.plot(x_values, func_values, label=f"Original {func_name}", color=color)
    plt.title(f"Original {func_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    freq_limit = 10

    # Define the frequency range
    frequencies = np.linspace(-freq_limit, freq_limit, 1000)
    sampled_times = x_values

    # Apply Fourier Transform to the function
    ft_data = fourier_transform(func_values, frequencies, sampled_times)
# Plot the frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2), color=color)
    plt.title(f"Frequency Spectrum of {func_name} (Frequency range -{freq_limit} to {freq_limit})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.show()

    ranges = [-10, -7.5, -2.5, 0, 2.5, 7.5, 10]

    reconstructed_func_values_2 = np.zeros(len(sampled_times))

    for i in range(len(ranges)):
        if(i < len(ranges) - 1):
            filtered_ft_data = np.copy(ft_data)
            filtered_ft_data[0][frequencies < ranges[i]] = 0
            filtered_ft_data[0][frequencies > ranges[i + 1]] = 0
            filtered_ft_data[1][frequencies < ranges[i]] = 0
            filtered_ft_data[1][frequencies > ranges[i + 1]] = 0

            # Plot the frequency spectrum
            plt.figure(figsize=(12, 6))
            plt.plot(frequencies, np.sqrt(filtered_ft_data[0]**2 + filtered_ft_data[1]**2), color=color)
            plt.title(f"Frequency Spectrum of {func_name} (Frequency range -{freq_limit} to {freq_limit})")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.show()

            # Reconstruct the function from FT data using Inverse Fourier Transform
            reconstructed_func_values = inverse_fourier_transform(filtered_ft_data, frequencies, sampled_times)

            # Plot the original and reconstructed functions for comparison
            plt.figure(figsize=(12, 4))
            # plt.plot(x_values, func_values, label=f"Original {func_name}", color=color)
            plt.plot(sampled_times, reconstructed_func_values,
                        label=f"Reconstructed {func_name} (Frequency range -{freq_limit} to {freq_limit})",
                        color="orange", linestyle="--")
            plt.title(f"Original vs Reconstructed {func_name} (Frequency range -{freq_limit} to {freq_limit})")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.show()

            reconstructed_func_values_2 += reconstructed_func_values
            
    
    # Plot the original and reconstructed functions for comparison
    plt.figure(figsize=(12, 4))
    plt.plot(x_values, func_values, label=f"Original {func_name}", color=color)
    plt.plot(sampled_times, reconstructed_func_values_2,
                label=f"Reconstructed {func_name} (Frequency range -{freq_limit} to {freq_limit})",
                color="orange", linestyle="--")
    plt.title(f"Original vs Reconstructed {func_name} (Frequency range -{freq_limit} to {freq_limit})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()     

    # Reconstruct the function from FT data using Inverse Fourier Transform
    reconstructed_func_values = inverse_fourier_transform(ft_data, frequencies, sampled_times)

    # Plot the original and reconstructed functions for comparison
    plt.figure(figsize=(12, 4))
    plt.plot(x_values, func_values, label=f"Original {func_name}", color=color)
    plt.plot(sampled_times, reconstructed_func_values,
                label=f"Reconstructed {func_name} (Frequency range -{freq_limit} to {freq_limit})",
                color="orange", linestyle="--")
    plt.title(f"Original vs Reconstructed {func_name} (Frequency range -{freq_limit} to {freq_limit})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()