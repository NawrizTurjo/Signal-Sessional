import numpy as np
import matplotlib.pyplot as plt
import os
class DiscreteSignal:
    def __init__(self, INF):
        self.INF = INF
        self.values = np.zeros(2 * INF + 1)  # Signal values are initialized to zero

    def set_value_at_time(self, time, value):
        index = time + self.INF  # Shift time to handle negative indices
        if 0 <= index < len(self.values):
            self.values[index] = value
        # else:
        #     raise ValueError("Time index is out of range.")

    def shift_signal(self, shift):
        # """Return a new signal instance with a shifted signal x[n - shift]."""
        new_signal = DiscreteSignal(self.INF)
        for i in range(-self.INF, self.INF + 1):
            index = i+shift
            if -self.INF <= index <= self.INF:
                new_signal.set_value_at_time(i, self.values[index + self.INF])
        return new_signal

    def add(self, other):
        # """Add two discrete signals and return the result."""
        if len(self.values) != len(other.values):
            raise ValueError("Signals must have the same length.")
        new_signal = DiscreteSignal(self.INF)
        new_signal.values = self.values + other.values
        return new_signal

    def multiply(self, other):
        # """Multiply two discrete signals element-wise and return the result."""
        if len(self.values) != len(other.values):
            raise ValueError("Signals must have the same length.")
        new_signal = DiscreteSignal(self.INF)
        new_signal.values = self.values * other.values
        return new_signal

    def multiply_const_factor(self, scaler):
        # """Multiply the signal by a constant factor and return the result."""
        new_signal = DiscreteSignal(self.INF)
        new_signal.values = self.values * scaler
        return new_signal

    def plot(self, ax, title="Discrete Signal",low=-1,high=4,filepath=None):
        time = np.arange(-self.INF, self.INF + 1)
        ax.stem(time, self.values, basefmt=" ")
        ax.set_title(title)
        ax.set_xlabel('n (Time Index)')
        ax.set_ylabel('x[n]')
        ax.set_ylim(low,high)
        ax.axhline(0, color='red', linestyle='-', linewidth=1.0)
        ax.grid(True)
        if filepath:
            plt.savefig(filepath)
            # print(f"Plot saved at: {filepath}")


class DiscreteLTI:
    def __init__(self,impulse_response):
        self.impulse_response = impulse_response
    def linear_combination_of_impulses(self,input_signal):
        impulses = []
        coefficients = []
        for n,value in enumerate(input_signal.values):
            # if value!=0:
            impulse = DiscreteSignal(input_signal.INF)
            impulse.set_value_at_time(n-input_signal.INF,value!=0)
            # impulse.plot()
            impulses.append(impulse)
            coefficients.append(value)
        # for i in range(-input_signal.INF,input_signal.INF+1):
        #     if input_signal[i+input_signal.INF]!=0:
        #         impulse = DiscreteSignal(input_signal.INF)
        #         impulse.set_value_at_time(i,1)
        #         # impulse.plot()
        #         impulses.append(impulse)
        #         coefficients.append(input_signal.values[i+input_signal.INF])
        return impulses,coefficients
    
    def output(self, input_signal):
        impulses, coefficients = self.linear_combination_of_impulses(input_signal)
        output_signal = DiscreteSignal(input_signal.INF)
        totalplots = 2*input_signal.INF+2
        rows = totalplots//3
        fig, axes = plt.subplots(rows,totalplots//rows, figsize=(10, 10))
        fig.suptitle("Response of Input Signal", fontsize=14)
        # loopi=0
        
        for i, (impulse, ax) in enumerate(zip(impulses, axes.flat[:-1])):  # Skip the last subplot for sum
            shifted_impulse = self.impulse_response.shift_signal(input_signal.INF - i)
            to_plot = shifted_impulse.multiply_const_factor(coefficients[i])
            title = f'h[n - ({i - input_signal.INF})]x[{i - input_signal.INF}]'
            to_plot.plot(ax, title,-1,input_signal.values.max()+2)

            output_signal = output_signal.add(to_plot)
            # loopi+=1
        
        # Plot the sum signal in the last subplot
        # print(loopi)
        sumt = f'Output = Sum'
        output_signal.plot(axes.flat[-1], sumt)

        plt.tight_layout()
        plt.savefig(os.path.join("discrete_signals", "Response_of_Input_Signal.png"))
        
        # plt.show()
        
        return output_signal
def main():

    # Stock Market Prices as a Python List
    # price_list = list(map(int, input("Stock Prices: ").split()))
    # n = int(input("Window size: "))
    # alpha = float(input("Alpha: "))

    # You may use the following input for testing purpose
    price_list = [10,11,12,9,10,13,15,16,17,18]
    n = 3
    alpha = 0.8

    # Determine the values after performing Exponential Smoothing
    # The length of exsm should be = len(price_list) - n + 1
    exsm = []

    print("Exponential Smoothing: " + ", ".join(f"{num:.2f}" for num in exsm))
    # Output should be: 11.68, 9.47, 9.82, 12.29, 14.40, 15.62, 16.64, 17.63

if __name__ == "__main__":
    main()