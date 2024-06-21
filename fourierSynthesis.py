import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import soundfile as sf

from plotting import *

def resample_audio(audio_path, target_sr=32000, duration=30):
    y, sr = librosa.load(audio_path, sr=target_sr, duration=30)
    return y, sr

def fft_on_wav(BASE_FOLDER, sample_song, sampling_rate=32000, segment_duration=0.5):
    song, sr = resample_audio(os.path.join(BASE_FOLDER, sample_song), target_sr=sampling_rate)

    print('song:', sample_song)
    print("Number of samples:", len(song))
    print("Sampling rate:", sr)
    print("Duration:", len(song)/sr, "seconds")

    segment_samples = int(segment_duration * sr) # Number of samples in each segment

    num_segments = len(song) // segment_samples

    chromosome = []

    for i in range(num_segments):

        start = i * segment_samples
        end = start + segment_samples
        segment = song[start:end]
        
        X = np.fft.fft(segment)
        magnitude_spectrum = np.abs(X)
        phase_values = np.angle(X)

        frame = [magnitude_spectrum, phase_values]
        chromosome.append(frame)

    print(f"peformed fft on {sample_song}.")
    return chromosome, song, sr

def contract_chromosome(chromosome, sr, song=None, waves_per_bin=25, phase_type='first_STTF'):
    avg_phases = {}
    final_mags = []
    first_phases = []
    for i,frame in enumerate(chromosome):
        mags = frame[0]
        phases = frame[1]
        magnitude_avg = 0
        if i == 0:
            first_phases = phases[0:(len(phases)//2) + waves_per_bin]
        mag_frame = []
        for i in range(((len(phases))//2) + waves_per_bin):
            if avg_phases.get(i):
                avg_phases[i] += phases[i]
            else:
                avg_phases[i] = phases[i]
            
            magnitude_avg += mags[i]

            if (i+1) % waves_per_bin == 0 and i != 0:
                magnitude_avg = float(magnitude_avg / waves_per_bin)
                mag_frame.append(magnitude_avg)
                magnitude_avg = 0
        final_mags.append(mag_frame)

    avg_phases = [avg_phases[i]/len(chromosome) for i in range(len(avg_phases))]
    avg_phases[0] = 0

    print(type(avg_phases), type(avg_phases[0]), type(final_mags), type(final_mags[0]), type(final_mags[0][0]))

    if phase_type == 'first_STTF':
        return [first_phases, final_mags]
    else:
        return [avg_phases, final_mags]

def ifft_on_contracted_chromosome(chromosome, sr, song=None, waves_per_bin=25,gen = 0,candidateIndex =0):
    concatenated_audio = np.array([])

    phases = chromosome[0]
    magnitude_chromosome = chromosome[1]

    for mag_frame in magnitude_chromosome:
    
        reconstructed_magnitude = np.array([value for i in range(len(mag_frame)) for value in [mag_frame[i]] * waves_per_bin])
        # print("len(full_magnitude):", len(reconstructed_magnitude))

        reconstructed_phase = np.array(phases)

        n = len(reconstructed_magnitude) - waves_per_bin

        # Reverse the reconstructed magnitude and phase values
        reconstructed_magnitude_reverse = reconstructed_magnitude[n-1:0:-1]
        reconstructed_phase_reverse = -reconstructed_phase[n-1:0:-1]

        reconstructed_magnitude = reconstructed_magnitude[0:n+1]
        reconstructed_phase = reconstructed_phase[0:n+1]

        full_magnitude = np.concatenate((reconstructed_magnitude, reconstructed_magnitude_reverse))
        full_phase = np.concatenate((reconstructed_phase, reconstructed_phase_reverse))

        X_reconstructed = full_magnitude * np.exp(1j * full_phase)
        
        inverse_X = np.fft.ifft(X_reconstructed)
        concatenated_audio = np.concatenate((concatenated_audio, inverse_X.real))

    import os
    if not os.path.exists("audio_output"):
        os.makedirs("audio_output")
    output_path = f"audio_output\\gen{gen}-candidate{candidateIndex}.wav"
    sf.write(output_path, concatenated_audio, sr, 'PCM_24')

def bestIFFT_on_contracted_chromosome(chromosome, sr, song=None, waves_per_bin=25,gen = 0,candidateIndex =0):
    concatenated_audio = np.array([])

    phases = chromosome[0]
    magnitude_chromosome = chromosome[1]

    for mag_frame in magnitude_chromosome:
    
        reconstructed_magnitude = np.array([value for i in range(len(mag_frame)) for value in [mag_frame[i]] * waves_per_bin])
        # print("len(full_magnitude):", len(reconstructed_magnitude))

        reconstructed_phase = np.array(phases)

        n = len(reconstructed_magnitude) - waves_per_bin

        # Reverse the reconstructed magnitude and phase values
        reconstructed_magnitude_reverse = reconstructed_magnitude[n-1:0:-1]
        reconstructed_phase_reverse = -reconstructed_phase[n-1:0:-1]

        reconstructed_magnitude = reconstructed_magnitude[0:n+1]
        reconstructed_phase = reconstructed_phase[0:n+1]

        full_magnitude = np.concatenate((reconstructed_magnitude, reconstructed_magnitude_reverse))
        full_phase = np.concatenate((reconstructed_phase, reconstructed_phase_reverse))

        X_reconstructed = full_magnitude * np.exp(1j * full_phase)
        
        inverse_X = np.fft.ifft(X_reconstructed)
        concatenated_audio = np.concatenate((concatenated_audio, inverse_X.real))

    import os
    if not os.path.exists("best_output"):
        os.makedirs("best_output")
    output_path = f"best_output\\BestInGen{gen}-{candidateIndex}.wav"
    sf.write(output_path, concatenated_audio, sr, 'PCM_24')

    # print(len(concatenated_audio), len(song))

    # plot_waveforms(song, concatenated_audio, sr, f"Original and Reconstructed Waveforms {waves_per_bin}-wpb")

if __name__ == "__main__":
    BASE_FOLDER = 'E:\\RESEARCH\\UFL Sound Genes\\sampling-rate\\dataset'
    sample_song = 'Shringar\\S_3.wav'

    '''
    phase_types = [
    'first_STTF', -> the phase values of the first Short Term Time Frame (STTF) are copied to all the other frames
    'avg_across_frames' -> the phase values are averaged based on frequency values, thereby rendering all the waves with 'x' frequency in all STTFs with an average phase value
    ]
    '''

    phase_type = 'first_STTF'

    segment_duration = 0.5  # Duration of each Short Term Time Frame (STTF) in seconds
    waves_per_bin = 4

    chromosome, song, sr = fft_on_wav(BASE_FOLDER, sample_song=sample_song, segment_duration=segment_duration)
    contracted_chromosome = contract_chromosome(chromosome, sr, song=None, waves_per_bin=waves_per_bin, phase_type=phase_type)

    ifft_on_contracted_chromosome(contracted_chromosome, sr=sr, song=song, waves_per_bin=waves_per_bin)