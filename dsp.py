import numpy as np
import matplotlib.pyplot as plt

def fir_convolve(M, h, x, n):
    result=0
    for k in range(0, M-1):
        result = h(k, M)*(x[n-k])
    return result

filter_cutoff = 1000

def filter_tf(n, M):
    wc = 2*np.pi*filter_cutoff
    return np.sin(wc*(n-(np.floor(M/2))))/(np.pi*(n-(np.floor(M/2))))

def fft(samples):
    # Perform FFT on the sampled signal
    fft_spectrum = np.fft.fft(samples)
    fft_frequencies = np.fft.fftfreq(len(samples), 1 / sampling_rate)

    # Take only the positive frequencies and normalize the FFT magnitude
    positive_freqs = fft_frequencies[:len(fft_frequencies)//2]
    fft_magnitude = np.abs(fft_spectrum[:len(fft_spectrum)//2]) / len(samples)
    
    return (positive_freqs, fft_magnitude)
    
    

# Parameters
frequency = 440  # Audio frequency in Hz (A4 pitch as an example)
frequency2 = 2000
sampling_rate = 44100  # Sampling rate in Hz (44.1 kHz)
amplitude = 1.0  # Amplitude of the sine wave
n_periods = 2  # Number of periods to plot


# Calculate the duration based on the number of periods
period = 1 / frequency  # Period of the sine wave (in seconds)
duration = n_periods * period  # Total duration of n periods

# Generate time axis for the continuous signal
t_continuous = np.linspace(0, duration, int(sampling_rate * duration * 10), endpoint=False)

# Generate time axis for the sampled signal
t_sampled = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate a complex sine wave for the continuous signal (440 Hz and 880 Hz components)
sine_wave_continuous = amplitude * np.sin(2 * np.pi * frequency * t_continuous) + 2 * amplitude * np.sin(2 * np.pi * frequency2 * t_continuous)

# Sample the continuous sine wave at the sampled time points
sine_wave_sampled = np.interp(t_sampled, t_continuous, sine_wave_continuous)

convolved = []

#convolve all of the samples with the filter impulse response
for n in range(0, len(sine_wave_sampled)):
    convolved.append(fir_convolve(len(sine_wave_sampled), filter_tf, sine_wave_sampled, n))

filtered = np.array(convolved)

positive_freqs, fft_magnitude = fft(sine_wave_sampled)

positive_freqs2, fft_magnitude2 = fft(filtered)


# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Time domain plot
ax1.plot(t_continuous, sine_wave_continuous, label='Continuous Sine Wave', color='blue')
ax1.stem(t_sampled, sine_wave_sampled, linefmt='r-', markerfmt='ro', basefmt=' ', label='Sampled Points', use_line_collection=True)
ax1.set_title(f'Sine Wave at {frequency} Hz and {2*frequency} Hz (Sampled at 44.1 kHz)\nPlotted over {n_periods} Periods')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.legend()
ax1.grid(True)

# Frequency domain plot (FFT)
ax2.plot(positive_freqs, fft_magnitude, color='green')
ax2.set_title('Frequency Domain Representation (FFT of Sampled Signal)')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_xscale('log')
ax2.set_ylabel('Magnitude')
ax2.grid(True)
ax2.set_xlim(0, 20000)  # Set frequency axis limit to 0-20,000 Hz

ax3.plot(positive_freqs2, fft_magnitude2, color='green')
ax3.set_title('Frequency Domain Representation (FFT of Filtered Signal)')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_xscale('log')
ax3.set_ylabel('Magnitude')
ax3.grid(True)
ax3.set_xlim(0, 20000)  # Set frequency axis limit to 0-20,000 Hz

plt.tight_layout()
plt.show()
