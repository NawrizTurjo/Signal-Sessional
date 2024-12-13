
import numpy as np
# Example usage
x = 65767879797907
y = 765454532435435345
# x=123
# y=456

def fft(x):

    N = len(x)
    
    # if N & (N - 1) != 0:
    #     next_power_of_2 = 1 << (N - 1).bit_length()
    #     # print(next_power_of_2 - N)
    #     x = np.pad(x, (0, next_power_of_2 - N))
    #     N = next_power_of_2

    # print(x)

    if N == 1:
        return x
    
    even = fft(x[::2])
    odd = fft(x[1::2])
    
    res_arr = np.zeros(N, dtype=complex)
    
    for k in range(N // 2):
        twiddle_factor = np.exp(-2j * np.pi * k / N)
        res_arr[k] = even[k] + twiddle_factor * odd[k]
        res_arr[k + N // 2] = even[k] - twiddle_factor * odd[k]

    return res_arr

def ifft(x):

    N = len(x)

    # if N & (N - 1) != 0:
    #     next_power_of_2 = 1 << (N - 1).bit_length() 
    #     x = np.pad(x, (0, next_power_of_2 - N))
    #     N = next_power_of_2

    # print(x)


    if N == 1:
        return x

    even = ifft(x[::2])
    odd = ifft(x[1::2])

    res_arr = np.zeros(N, dtype=complex)

    for k in range(N // 2):
        twiddle_factor = np.exp(2j * np.pi * k / N) 
        res_arr[k] = even[k] + twiddle_factor * odd[k]
        res_arr[k + N // 2] = even[k] - twiddle_factor * odd[k]

    return res_arr / 2


def cross_correlation_fft(signal_A, signal_B):
    # dft_A = fft(signal_A)
    # dft_B = fft(signal_B)
    
    # cross_corr_freq = dft_A * np.conj(dft_B)
    # cross_corr_freq = dft_B * np.conj(dft_A)
    cross_corr_freq = signal_B * np.conj(signal_A)
    
    cross_corr_time = ifft(cross_corr_freq)
    cross_corr_time = np.roll(cross_corr_time, len(cross_corr_time) // 2)
    return np.real(cross_corr_freq)

def pad_extra(x,mxsize,y):
    N = len(x)
    M = len(y)
    
    # if mxsize & (mxsize - 1) != 0:
    next_power_of_2 = 1 << (mxsize - 1).bit_length()
    x = np.pad(x, (0, next_power_of_2 - N))
    y = np.pad(y, (0, next_power_of_2 - M))
    mxsize = next_power_of_2
    print(x)
    print(y)
    return x,y


#converting to digit arrays(discrete signal)
x_digits = [int(digit) for digit in str(x)]

y_digits = [int(digit) for digit in str(y)]

xln = len(x_digits)
yln = len(y_digits)

print(xln,yln)

# print(x_digits)
# print(y_digits)
# x_main = x_digits
# y_main = y_digits

mxsize = (len(x_digits)+len(y_digits)-1)
print(mxsize)

x_digits,y_digits = pad_extra(x_digits,mxsize,y_digits)
# pad_extra(x_main)
# pad_extra(y_main)
# print(len(x_digits))

x_digits = fft(x_digits)
y_digits = fft(y_digits)
# x_digits = ifft(x_digits)
# y_digits = ifft(y_digits)

# print(x_digits)
# print(y_digits)



ans = x_digits*(y_digits)
# ans_len =  max(len(x_digits),len(y_digits))+1
ans_len = mxsize
actual_len = mxsize
print(actual_len)
# print(ans_len)
# print(np.real(ans))
ans = ifft(ans)

print(np.real(ans))

# cross = cross_correlation_fft(x_digits,y_digits)

ans_real = np.round(np.real(ans))

res_arr = []

carry = 0
# print(ans_real)

for i in range(ans_len):
    mod = int(ans_real[ans_len-i-1] % 10)
    carry = int(ans_real[ans_len-i-1] / 10)
    ans_real[ans_len-2-i] += carry
    # print(mod)
    # print(carry)
    res_arr.append(mod)

while carry > 0:
    mod = carry % 10
    carry = carry // 10
    res_arr.append(mod)

print(res_arr)

value = 0
for i in range(len(res_arr)):
    value += res_arr[i] * (10**i)

print(value)





# print(cross)

