import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import trapz

    

# Load and preprocess the image
image = plt.imread('noisy_image.png')  # Replace with your image file path
# show the image
plt.figure()
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.show()

if image.ndim == 3:
    image = np.mean(image, axis=2)  # Convert to grayscale

image = image / 255.0  # Normalize to range [0, 1]
print (image.shape)

sample_rate = 1000 

def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    for i, freq in enumerate(frequencies):
        cosine_part = np.cos(2 * np.pi * freq * sampled_times)
        sine_part = np.sin(2 * np.pi * freq * sampled_times)
        
        # Real part
        ft_result_real[i] = np.trapezoid(signal * cosine_part, sampled_times,num_freqs)
        # Imaginary part
        ft_result_imag[i] = -np.trapezoid(signal * sine_part, sampled_times,num_freqs)
        
    return ft_result_real, ft_result_imag

def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    
    for i, time in enumerate(sampled_times):
        cosine_part = np.cos(2 * np.pi * frequencies * time)
        sine_part = np.sin(2 * np.pi * frequencies * time)
        real_part_sum = np.trapezoid(ft_signal[0] * cosine_part, frequencies, num_freqs)
        imag_part_sum = np.trapezoid(ft_signal[1] * sine_part, frequencies, num_freqs)
        
        # Sum the real and imaginary parts for each time point
        reconstructed_signal[i] = real_part_sum - imag_part_sum
    
    return reconstructed_signal

# frequencies = np.linspace(0, sample_rate / 2, num=image.shape[1])
frequencies = np.linspace(0, sample_rate, 64)
# Set parameters for interval sampling and FT
interval_step = 1  # Adjust for sampling every 'interval_step' data points  
data_sampled = image[::interval_step]
max_time = len(data_sampled) / (sample_rate / interval_step)
sampled_times = np.linspace(0, max_time, num=len(data_sampled))

# Apply FT with trapezoidal integration
# ft_data = fourier_transform(data_sampled, frequencies, sampled_times)

# ft = image.copy()

# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         ft_data = fourier_transform(image[i][j], frequencies, sampled_times)
# reconstructed_image = np.zeros_like(image.shape[1])
# for i in range(image.shape[1]):
#     ft_data = fourier_transform(image[i], frequencies, sampled_times)
    
    # filtered_ft_data = np.zeros((2, image.shape[1]))
    # plt.title("Frequency Spectrum of the Audio Signal (Custom FT with Trapezoidal Integration)")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude")
    # plt.show()
    # filtered_ft_data= np.zeros((2, 1000, 64))
    # filtered_ft_data[0] = ft_data[0].copy()
    # filtered_ft_data[1] = ft_data[1].copy()
    # print(ft_data.count)


    # plt.figure(figsize=(12, 6))
    # plt.plot(frequencies, np.sqrt(filtered_ft_data[0]**2 + filtered_ft_data[1]**2))
    # plt.title("Filtered Frequency Spectrum (Unwanted Frequencies Removed)")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude")
    # plt.show()

    # reconstructed_image[i] = inverse_fourier_transform(filtered_ft_data, frequencies, sampled_times)
    # plt.figure(figsize=(12, 4))
    # plt.plot(sampled_times, reconstructed_image[i])
    # plt.title("Reconstructed (Denoised) Audio Signal (Time Domain)")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.show()

# for 0,0
denoised_image = image.copy()
# denoised_image = np.zeros((64,64))
for i in range(len(image)):
    ft_data = fourier_transform(image[i], frequencies, sampled_times)
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
    plt.title("Frequency Spectrum of the Audio Signal (Custom FT with Trapezoidal Integration)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    # plt.show()
    
    filtered_ft_data= np.zeros((2, 64))
    filtered_ft_data[0] = ft_data[0].copy()
    filtered_ft_data[1] = ft_data[1].copy()

    filtered_ft_data[0] = np.where((frequencies < 80),0, filtered_ft_data[0])
    filtered_ft_data[0] = np.where((frequencies > 910),0, filtered_ft_data[0])
    filtered_ft_data[1] = np.where((frequencies < 80),0, filtered_ft_data[1])
    filtered_ft_data[1] = np.where((frequencies > 910),0, filtered_ft_data[0])

    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, np.sqrt(filtered_ft_data[0]**2 + filtered_ft_data[1]**2))
    plt.title("Filtered Frequency Spectrum (Unwanted Frequencies Removed)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    # plt.show()

    denoised_image[i] = inverse_fourier_transform(filtered_ft_data, frequencies, sampled_times)
    plt.figure(figsize=(12, 4))
    plt.plot(sampled_times, denoised_image[i])
    plt.title("Reconstructed (Denoised) Audio Signal (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    # plt.show()



plt.imsave('denoised_image.png', denoised_image, cmap='gray')


plt.figure()
plt.title('Denoised Image')
plt.imshow(denoised_image, cmap='gray')
# plt.show()
