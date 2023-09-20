import numpy as np
import random
import numpy as np
import wave
import time
import sys
from tqdm import tqdm

# Function to decode the chromosome into audio and save it to a wave file
def decode(chromosome, filename):
    # Enlarge the chromosome to add more waves to each bin
    enlargedChromosome = enlarge_chromosome(chromosome, waves_per_bin=3, min_frequency=400, max_frequency=7000)

    # Record the start time for execution
    start = time.time()

    # Constants related to audio frames and bins
    frame_duration = 0.2  # seconds
    frames_per_sec = 44100
    waves_per_bin = 20

    num_frames = len(enlargedChromosome)
    num_samples = int(frames_per_sec * frame_duration * num_frames)

    # Initialize an array to hold the waveform samples
    waveform = np.zeros(num_samples, dtype=np.int16)

    # tqdm is a component from the tqdm library that helps us see the progress bar
    with tqdm(total=len(enlargedChromosome), desc="Converting each frame to audio") as pbar:
        for frame_idx, frame in enumerate(enlargedChromosome):
            pbar.update(1)
            for bin_idx, bin in enumerate(frame):
                for i, wave in enumerate(bin):
                    amplitude = wave[0]
                    phase = 0.0
                    frequency = wave[1]
                    bin_samples = generate_bin_samples(amplitude, phase, frequency, frame_duration, frames_per_sec, waves_per_bin)
                    segment_start = frame_idx * int(frame_duration * frames_per_sec)
                    segment_end = (frame_idx + 1) * int(frame_duration * frames_per_sec)
                    waveform[segment_start:segment_end] += bin_samples

    # Print information about the waveform and chromosome
    print(waveform.shape)
    print(np.array(enlargedChromosome).shape)

    # Write the waveform to a wave file
    write_wave(filename, waveform)
    print("Decoding completed!")
    print(f"Output Audio Filename: {filename}")
    print("The Decoding Execution Time is:", (time.time() - start), "s")

def generate_bin_samples(amplitude, phase, frequency, duration, frames_per_sec, waves_per_bin):
    # This line creates a time array t that starts from 0 and goes up to the specified duration.
    # The linspace function is used to evenly divide this time range into a sequence of time points.
    # The endpoint=False argument indicates that the endpoint (the final time point) should not be included in the sequence.
    t = np.linspace(0, duration, int(duration * frames_per_sec), endpoint=False)

    # Here, the frequency array is created to match the time array t. 
    # The np.full_like function fills an array with the specified frequency value, creating an array of the same shape as t.
    # This is done because the frequency can be constant for the entire duration of the bin.

    frequency = np.full_like(t, frequency)  # Reshape frequency to match t

    # This line generates the waveform samples. It uses the sine function (np.sin) to create a sinusoidal waveform.
    # The formula 2 * np.pi * frequency * t + np.deg2rad(phase) calculates the instantaneous phase of the waveform as it evolves over time.
    # The amplitude scales the sine wave to the desired level.

    waveform = amplitude * np.sin(2 * np.pi * frequency * t + np.deg2rad(phase))

    # Finally, the generated waveform is scaled to fit within the range of 16-bit signed integers, which is the format typically used in audio files (WAV format).
    # The multiplication by 32767 scales the waveform to the range [-32767, 32767], and astype(np.int16) converts it to a NumPy array with a data type of 16-bit integers.
    # This prepares the waveform for writing to an audio file.

    return (waveform * 32767).astype(np.int16)


def write_wave(filename, waveform):
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(44100)
        wav_file.writeframes(waveform.tobytes())

# Function to generate a random chromosome based on specified parameters
def generate_chromosome(Cl=300, Gl=50, A=1.0):
# Cl = Number of frames
# Gl = Number of bins per frame
# A = Amplitude Range

# Random Chromosome Generator
# Creates a random chromosome with the permissible boundary values

    L=[]

    for i in range(Cl):
        L.append([])
        
        for j in range(Gl):
            L[i].append([random.uniform(0,A), random.uniform(0,360)])
    
    return L

# Function to enlarge the chromosome by adding more waves to each bin
# This version is compatible with the DE (Differential Evolution) code
def enlarge_chromosome(chromosome, waves_per_bin=20, min_frequency=20.0, max_frequency=20020.0):
    # Enlarge the chromosome by adding more waves to each bin
    frequency_Range = max_frequency - min_frequency
    current_frequency = min_frequency
    frequency_difference = frequency_Range / 1000

    for frame in chromosome:
        current_frequency = min_frequency
        for i in range(len(frame)):
            if current_frequency != min_frequency:
                current_frequency += frequency_difference
            frame[i] = [frame[i]]
            frame[i][0].append(current_frequency)
            for _ in range(waves_per_bin - 1):
                current_frequency += frequency_difference
                frame[i].append([frame[i][0][0], frame[i][0][1], current_frequency])
    return chromosome                                           


if __name__ == "__main__":
    chromosome = []
    audio_file_name = ""

    num_arguments = len(sys.argv)

    if num_arguments == 2:
        audio_file_name = sys.argv[1]
    elif num_arguments == 3:
        chromosome = sys.argv[1]
        audio_file_name = sys.argv[2]
    elif num_arguments == 1:
        audio_file_name = "sample_output"
    
    chromosome = generate_chromosome()

    if len(audio_file_name) < 4 or audio_file_name[-4] != ".wav" :
        audio_file_name += ".wav"
    decode(chromosome, audio_file_name)

