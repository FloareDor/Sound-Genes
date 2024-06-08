import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import soundfile as sf
import random

def plot_waveform(inverse_signal, sr, title):
    plt.figure(figsize=(18, 5))
    t = np.linspace(0, len(inverse_signal) / sr, len(inverse_signal))
    plt.plot(t, inverse_signal.real)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()

def plot_waveforms(original_signal, reconstructed_signal, sr, title):
    plt.figure(figsize=(18, 5))
    t = np.linspace(0, len(original_signal) / sr, len(original_signal))
    plt.plot(t, original_signal, color='blue', label='Original Waveform')
    plt.plot(t, reconstructed_signal.real, color='red', label='Reconstructed Waveform')
    
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_magnitude_spectrum(signal, sr, title, f_ratio=1):
    X = np.fft.fft(signal)
    X_mag = np.abs(X)
   
    plt.figure(figsize=(18, 5))
   
    f = np.linspace(0, sr, len(X_mag))
    f_bins = int(len(X_mag)*f_ratio)  
   
    plt.plot(f[:f_bins], X_mag[:f_bins])
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
    

def plot_magnitude_spectrum2(signal, sr, title, segment_duration, f_ratio=1):
    X = np.fft.fft(signal)
    X_mag = np.abs(X)
   
    plt.figure(figsize=(18, 5))
   
    f = np.linspace(0, sr, len(X_mag))
    f_bins = int(len(X_mag)*f_ratio)  
   
    plt.plot(range(f_bins), X_mag[:f_bins])
    plt.xlabel(f'Frequency (for a packet of - {segment_duration} sec)')
    plt.ylabel(f'Amplitude')
    plt.title(title)

    # plt.xticks(range(0, len(X_mag)), 1)

def plot_magnitude_spectrum_bar(signal, sr, title, segment_duration, f_ratio=1):
    X = np.fft.fft(signal)
    X_mag = np.abs(X)
   
    plt.figure(figsize=(18, 5))
   
    f = np.linspace(0, sr, len(X_mag))
    f_bins = int(len(X_mag)*f_ratio)  
   
    plt.stem(range(len(X_mag)), X_mag[:f_bins], basefmt=' ')
    plt.xlabel(f'Frequency (for a packet of - {segment_duration} sec)')
    plt.ylabel(f'Amplitude')
    plt.title(title)
    plt.xticks(range(0, len(X_mag), 1))

def plot_phase_spectrum(signal, sr, title, f_ratio=1):
    X = np.fft.fft(signal)
    X_phase = np.angle(X)
   
    plt.figure(figsize=(18, 5))
   
    f = np.linspace(0, sr, len(X_phase))
    f_bins = int(len(X_phase)*f_ratio)  
   
    plt.plot(f[:f_bins], X_phase[:f_bins])
    plt.xlabel('Frequency (Hz)')
    plt.title(title)

def plot_sampling_points(segment, samplerate=32000):
    N = len(segment)   # number of samples
    t = np.arange(N)/samplerate     # time array

    plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
    plt.plot(t, segment, 'b-')   # Plot the signal as a blue line
    plt.plot(t, segment, 'ro')   # Plot each sample point as a red dot

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Signal with Sample Points')
    plt.grid(True)
    plt.show()