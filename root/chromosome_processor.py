import numpy as np
import random
import numpy as np
import wave
import time
import sys
from tqdm import tqdm

def decode(chromosome, filename):
    chromosome = enlarge_chromosome2(chromosome, waves_per_bin=20, min_frequency=20.0, max_frequency=20020.0)
    start = time.time()
    frame_duration = 0.2  # seconds
    frames_per_sec = 44100
    num_bins = 50
    waves_per_bin = 20
    sample_width = 2  # in bytes (16-bit)

    num_frames = len(chromosome)
    num_samples = int(frames_per_sec * frame_duration * num_frames)

    # Initialize an array to hold the waveform samples
    waveform = np.zeros(num_samples, dtype=np.int16)
    k = 0
    with tqdm(total=len(chromosome), desc="Converting each frame to audio") as pbar:
        for frame_idx, frame in enumerate(chromosome):
            # if frame_idx % 33:
            #     print(f"{frame_idx}/{len(chromosome)}")
            # print(frame)
            pbar.update(1)
            for bin_idx, bin in enumerate(frame):
                # print(bin)
                for i,wave in enumerate(bin):
                    amplitude = wave[0]
                    phase = wave[1]
                    frequency = wave[2]
                    # if i == 18:
                    #     if bin_idx == 3:
                    #         amplitude = bin[0][0]
                    bin_samples = generate_bin_samples(amplitude, phase, frequency, frame_duration, frames_per_sec, waves_per_bin)
                    segment_start = frame_idx * int(frame_duration * frames_per_sec)
                    segment_end = (frame_idx + 1) * int(frame_duration * frames_per_sec)
                    waveform[segment_start:segment_end] += bin_samples

    print(waveform.shape)
    print(np.array(chromosome).shape)

    write_wave(filename, waveform)
    print("Decoding completed!")
    print("The Decoding Execution Time is :", (time.time()-start), "s")

def generate_bin_samples(amplitude, phase, frequency, duration, frames_per_sec, waves_per_bin):
    t = np.linspace(0, duration, int(duration * frames_per_sec), endpoint=False)
    frequency = np.full_like(t, frequency)  # Reshape frequency to match t
    waveform = amplitude * np.sin(2 * np.pi * frequency * t + np.deg2rad(phase))
    return (waveform * 32767).astype(np.int16)


def write_wave(filename, waveform):
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(44100)
        wav_file.writeframes(waveform.tobytes())

# def generate_chromosome():
#     num_bins = 50
#     waves_per_bin = 20
#     num_frames = 300
#     min_frequency = 20.0    
#     max_frequency = 20020.0
#     frequency_Range = max_frequency - min_frequency
#     current_frequency = min_frequency
#     frequency_difference = frequency_Range/1000
#     chromosome = []

#     for frame_idx in range(num_frames):
#         frame = []
#         current_frequency = min_frequency
#         for bin_idx in range(num_bins):
#             bin_waves = []
#             amplitude = random.uniform(-1.0, 1.0)
#             phase = random.uniform(0.0, 360.0)
#             for wave_idx in range(waves_per_bin):
#                 bin_waves.append(amplitude)
#                 bin_waves.append(phase)
#                 bin_waves.append(current_frequency)
#                 current_frequency+=frequency_difference

#             frame.append(bin_waves)

#         chromosome.append(frame)

#     return chromosome


def generate_chromosome(frames = 300, bins=50, amplitude_range=1):
    # Generate a chromosome with single waves per bin
    chromosome = []

    for _ in range(frames):
        frame = []
        for _ in range(bins):
            bin = []
            amplitude = random.uniform(-amplitude_range,amplitude_range)
            phase = random.uniform(0.0, 360.0)
            # amplitude = 0.5
            # phase = 0.0
            bin.append([amplitude, phase])
            frame.append(bin)

        chromosome.append(frame)
   
    return chromosome

def generate_chromosome2(Cl=300, Gl=50, A=1.0):
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

### this version of the enlarge function works perfectly with the DE code
def enlarge_chromosome2(chromosome, waves_per_bin=20, min_frequency=20.0, max_frequency=20020.0):
    # Enlarge the chromosome by adding more waves to each bin

    frequency_Range = max_frequency - min_frequency
    current_frequency = min_frequency
    frequency_difference = frequency_Range / 1000

    for frame in chromosome:
        current_frequency = min_frequency
        for i in range(len(frame)):
            if current_frequency != min_frequency:
                current_frequency+=frequency_difference
            frame[i] = [frame[i]]
            frame[i][0].append(current_frequency)
            for _ in range(waves_per_bin-1):
                current_frequency += frequency_difference
                frame[i].append([frame[i][0][0], frame[i][0][1], current_frequency])
    return chromosome

# def enlarge_chromosome(chromosome, waves_per_bin=20, min_frequency=20.0, max_frequency=20020.0):
#     # Enlarge the chromosome by adding more waves to each bin

#     frequency_Range = max_frequency - min_frequency
#     current_frequency = min_frequency
#     frequency_difference = frequency_Range / 1000

#     for frame in chromosome:
#         current_frequency = min_frequency
#         for bin in frame:
#             if current_frequency != min_frequency:
#                 current_frequency+=frequency_difference
#             bin[0].append(current_frequency)
#             for _ in range(waves_per_bin-1):
#                 current_frequency += frequency_difference
#                 bin.append([bin[0][0], bin[0][1], current_frequency])
#         # print(frame)

#     return chromosome

# def generate_full_chromosome(frames = 300, bins=50, amplitude_range=1, waves_per_bin=20, min_frequency=20.0, max_frequency=20020.0):
#     return enlarge_chromosome(generate_chromosome(frames=frames, bins=bins, amplitude_range=amplitude_range), waves_per_bin=waves_per_bin, min_frequency=min_frequency, max_frequency=max_frequency)


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
    
    chromosome = generate_chromosome2()

    if len(audio_file_name) < 4 or audio_file_name[-4] != ".wav" :
        audio_file_name += ".wav"
    decode(chromosome, audio_file_name)
