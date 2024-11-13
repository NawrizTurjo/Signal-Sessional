import numpy as np
import matplotlib.pyplot as plt

# import os


# save_folder = "Parabolic_Wave"
# os.makedirs(save_folder, exist_ok=True)

# Define the interval and function and generate appropriate x values and y values
# Define the interval for x and the function y = x^2
x_values = np.linspace(-10, 10, 1000)
y_values = np.where(np.abs(x_values) <= 2, x_values**2, 0) ## parabolic
# y_values = np.where(np.abs(x_values) <= 2, 1, 0) ## square wave
# y_values = np.where(np.abs(x_values)<=2,0.5*np.abs(4*(2/8) * np.abs(((x_values+2)-(8/4))%(8/2))-2),0) ## Triangular
# y_values = np.where(np.abs(x_values)<=2,4* (x_values/(4)-np.floor(0.5+x_values/(4))),0) ## Sawtooth


# Plot the original function
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original y = x^2")
plt.title("Original Function (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
plt.legend()
plt.show()

# plt.savefig(os.path.join(save_folder, "original_function.png"))  # Save as PNG
# plt.close()


# Define the sampled times and frequencies
sampled_times = x_values
frequencies = np.linspace(-5, 5, 1000)  # Example frequency range; you may adjust this as needed

# Fourier Transform 
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    for i, freq in enumerate(frequencies):
        cosine_part = np.cos(2 * np.pi * freq * sampled_times)
        sine_part = np.sin(2 * np.pi * freq * sampled_times)
        
        # Real part
        ft_result_real[i] = np.trapezoid(signal * cosine_part, sampled_times)
        # Imaginary part
        ft_result_imag[i] = -np.trapezoid(signal * sine_part, sampled_times)
        
    return ft_result_real, ft_result_imag


# Apply FT to the sampled data
ft_data = fourier_transform(y_values, frequencies, sampled_times)
#  plot the FT data
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
plt.title("Frequency Spectrum of y = x^2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()

# plt.savefig(os.path.join(save_folder, "frequency_spectrum_1.png"))  # Save as PNG
# plt.close()


# Inverse Fourier Transform 
def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    
    for i, time in enumerate(sampled_times):
        cosine_part = np.cos(2 * np.pi * frequencies * time)
        sine_part = np.sin(2 * np.pi * frequencies * time)
        real_part_sum = np.trapezoid(ft_signal[0] * cosine_part, frequencies)
        imag_part_sum = np.trapezoid(ft_signal[1] * sine_part, frequencies)
        
        # Sum the real and imaginary parts for each time point
        reconstructed_signal[i] = real_part_sum - imag_part_sum
    
    return reconstructed_signal


# Reconstruct the signal from the FT data
reconstructed_y_values = inverse_fourier_transform(ft_data, frequencies, sampled_times)
# Plot the original and reconstructed functions for comparison
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original y = x^2", color="blue")
plt.plot(sampled_times, reconstructed_y_values, label="Reconstructed y = x^2", color="red", linestyle="--")
plt.title("Original vs Reconstructed Function (y = x^2)")
plt.xlabel("x")
# plt.xlim(-10, 10)
plt.ylabel("y")
plt.legend()
plt.show()

# plt.savefig(os.path.join(save_folder, "original_vs_reconstructed_1.png"))  # Save as PNG
# plt.close()
