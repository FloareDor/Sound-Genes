import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import soundfile as sf
import random
from plotting import *

def average_down_the_phases(chromosome):
    """
    Calculates the average phase values across all frames for each frequency index.
    
    Args:
        chromosome: A list of frames, where each frame is a list of bins containing magnitude and phase values.
        
    Returns:
        A dictionary mapping frequency indices to their average phase values across all frames.
    """
    avg_phases= {}
    for frame in chromosome:
        # Un-bin the magnitude and phase values
        reconstructed_magnitude = []
        reconstructed_phase = []
        for bin in frame:
            reconstructed_magnitude.extend([value[0] for value in bin])
            reconstructed_phase.extend([value[1] for value in bin])
        for i in range(len(reconstructed_phase)):
            if avg_phases.get(i):
                avg_phases[i] += reconstructed_phase[i]
            else:
                avg_phases[i] = reconstructed_phase[i]
    for index in avg_phases:
        avg_phases[index] /= len(chromosome)
    print(len(avg_phases), len(avg_phases), len(avg_phases))
    return avg_phases

def resample_audio(audio_path, target_sr=32000):
    """
    Resamples the audio file to the target sampling rate.
    
    Args:
        audio_path: The path to the audio file.
        target_sr: The target sampling rate (default is 32000).
        
    Returns:
        A tuple containing the resampled audio data and the target sampling rate.
    """
    y, sr = librosa.load(audio_path, sr=target_sr, duration=30)
    return y, sr

def wav_to_chromosome(BASE_FOLDER, sample_song, phase_type="original", sampling_rate=32000, segment_duration=0.5, waves_per_bin=25):
    """
    Converts a WAV file to a chromosome representation.
    
    Args:
        BASE_FOLDER: The base folder containing the audio files.
        sample_song: The name of the audio file.
        phase_type: The type of phase processing to apply (default is "original").
        sampling_rate: The target sampling rate (default is 32000).
        segment_duration: The duration of each segment in seconds (default is 0.5).
        waves_per_bin: The number of waves per bin (default is 25).
        
    Returns:
        A tuple containing the chromosome representation, the original audio data, and the sampling rate.
    """

    song, sr = resample_audio(os.path.join(BASE_FOLDER, sample_song), target_sr=sampling_rate)

    print('song:', sample_song)
    print("Number of samples:", len(song))
    print("Sampling rate:", sr)
    print("Duration:", len(song)/sr, "seconds")

    segment_samples = int(segment_duration * sr) # Number of samples in each segment

    num_segments = len(song) // segment_samples

    chromosome = []

    phase_room = []

    for i in range(num_segments):

        start = i * segment_samples
        end = start + segment_samples
        segment = song[start:end]

        
        X = np.fft.fft(segment)
        magnitude_spectrum = np.abs(X)
        phase_values = np.angle(X)
        
        frame = []
        bin = []
        magnitude_sum = 0
        phase_sum = 0
        for j in range(len(magnitude_spectrum)):
            phase_room.append(phase_values[j])

            if i == 0:
                phase_room.append(phase_values[j])

            magnitude_sum += magnitude_spectrum[j]
            phase_sum += phase_values[j]
            if (j+1) % waves_per_bin == 0 and j != 0:
                magnitude_avg = magnitude_sum / waves_per_bin
                phase_avg = phase_sum / waves_per_bin
                magnitude_sum = 0
                phase_sum = 0
                
                # bin = [[magnitude_spectrum[k], phase_values[k]]for k in range(j-waves_per_bin+1, j+1)] # original magnitudes and phases
                if phase_type == "original" or phase_type == "avg_across_frames":
                    bin = [[magnitude_avg, phase_values[k]]for k in range(j-waves_per_bin+1, j+1)] # avg magnitudes per bin, same phase values
                if phase_type == "first_STTF":
                    bin = [[magnitude_avg, phase_room[k]] for k in range(j-waves_per_bin+1, j+1)]
                if phase_type == "avg_per_frame":
                    bin = [[magnitude_avg, phase_avg]for _ in range(j-waves_per_bin+1, j+1)]
                frame.append(bin)
                bin = []

        frame.append(bin)
        bin = []
    
    return chromosome, song, sr
    
def ifft_on_chromosome(chromosome, sr, song=None, phase_type="original"):
    """
    Performs Inverse Fast Fourier Transform (IFFT) on the chromosome representation to reconstruct the audio.
    
    Args:
        chromosome: The chromosome representation of the audio.
        sr: The sampling rate of the audio.
        song: The original audio data (default is Non        phase_type: The type of phase processing applied (default is "original").
        
    Returns:
        None
    """
    concatenated_audio = np.array([])
    if phase_type == "avg_across_frames":
        avg_phases = average_down_the_phases(chromosome=chromosome)
    for frame in chromosome:
        # Un-bin the magnitude and phase values
        reconstructed_magnitude = []
        reconstructed_phase = []
        for bin in frame:
            reconstructed_magnitude.extend([value[0] for value in bin])
            reconstructed_phase.extend([value[1] for value in bin])
        
        # Reverse the reconstructed magnitude and phase values
        # reconstructed_magnitude_reverse = reconstructed_magnitude[::-1]
        # reconstructed_phase_reverse = reconstructed_phase[::-1]

        # Reversing does not work properly, so I just made it empty.
        reconstructed_magnitude_reverse = []
        reconstructed_phase_reverse = []

        if phase_type == "avg_across_frames":
            reconstructed_phase = []
            for ind in avg_phases:
                reconstructed_phase.append(avg_phases[ind])

        # Concatenate the reconstructed and reversed values
        full_magnitude = np.concatenate((reconstructed_magnitude, reconstructed_magnitude_reverse))
        full_phase = np.concatenate((reconstructed_phase, reconstructed_phase_reverse))

        # Reconstruct the complex numbers for IFFT
        X_reconstructed = full_magnitude * np.exp(1j * full_phase)
        
        inverse_X = np.fft.ifft(X_reconstructed)
        concatenated_audio = np.concatenate((concatenated_audio, inverse_X.real))
        
    output_path = f"output-avg_magnitudes-avg-phases-across-frames-wpb-{waves_per_bin}.wav"

    sf.write(output_path, concatenated_audio, sr, 'PCM_24')

    print(len(concatenated_audio), len(song))

    plot_waveforms(song, concatenated_audio, sr, f"Original and Reconstructed Waveforms {waves_per_bin}-wpb")


if __name__ == "__main__":
    BASE_FOLDER = 'E:\\RESEARCH\\UFL Sound Genes\\sampling-rate\\dataset'
    sample_song = 'Shringar\\S_9.wav'

    '''
    phase_types = [
    'original', -> the phase values will remain unchanged
    'first_STTF', -> the phase values of the first Short Term Time Frame (STTF) are copied to all the other frames
    'avg_per_frame', -> the phase values of all the waves in each STTF are set to the average phase value in that STTF
    'avg_across_frames' -> the phase values are averaged based on frequency values, thereby rendering all the waves with 'x' frequency in all STTFs with an average phase value
    ]
    '''

    segment_duration = 0.5  # Duration of each Short Term Time Frame (STTF) in seconds
    waves_per_bin = 25
    phase_type = 'original'

    chromosome, song, sr = wav_to_chromosome(BASE_FOLDER, sample_song, phase_type=phase_type)
    ifft_on_chromosome(chromosome, sr, song, phase_type=phase_type)
    print("Done!")
