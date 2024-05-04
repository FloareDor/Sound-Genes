### Revamped the whole code to make it modular, standardised and made the bin frequency
### mapping more direct, rather that using a hit or miss approach

import numpy as np
 
from scipy.io import wavfile

from scipy.fft import ifft

# My entire understanding of fft and hence ifft comes from this source:-
# http://howthefouriertransformworks.com/understanding-the-output-of-an-fft/

def decode(Lold, index, generation, Minfrq, Maxfrq, Cl, Gl, Wpb, TS, Srate):
    
# Arguments explaination:-
# 1) Lold: This is the chromosome returned by DE. It does not contain the frequencies
#   nor corresponding to the amplitudes and phases
# 2) index: This is the index of the chromosome in each generation, used for file naming
# 3) generation: This is the generation number of the chromosome, used for file naming
# 4) Minfrq: This is the minimum frequency
# 5) Maxfrq: This is the maximum frequency
# 6) Cl: (Chromosome Length) This is the number of frames
# 7) Gl: (Gene Length) This is the number of bins in a frame
# 8) Wpb: (Waves Per Bin) This is the number of waves in a bin that have the same amplitude, phase
# 9) TS: (Total Samples) This is the number of samples in the entire created audio file
# 10) Srate: (Sample Rate) This is the number of samples per second


    L=expand(Lold=Lold, Minfrq=Minfrq, Maxfrq=Maxfrq, Cl=Cl, Gl=Gl, Wpb=Wpb)
    # Now every bin contains [Amplitude, Phase, Frequency]
    

    Flen=TS/(Cl*Srate)
    # Frame Length
    # This is the length of a timeframe in seconds

    SperF=int(Srate*Flen)
    # Samples per Frame
    # This is the length of a frame, in samples not seconds

    Out=np.zeros(TS, dtype=np.int16)
    # This is the array that will be finally converted to a wav


    for Fn in range(Cl):
    # For Frame Number in the chromosome
    # For every frame

        Wf=frame_to_wav(F=L[Fn], Srate=Srate, SperF=SperF)

        if Fn==Cl-1:
        # Edge case if the song is not perfectly divisible by the frame length

            for i in range(SperF):
                if (Fn*SperF+i)<TS:
                    Out[Fn*SperF+i]=Wf[i]

        else:

            for i in range(SperF):
                Out[Fn*SperF+i]=Wf[i]
            # Add the frame to the Out array
            # Here Fn*SperF gives the time point of the start of this frame


    wavfile.write(f"audio_output/gen{generation}-{index}.wav", Srate, Out)
    # Write the wavfile

    return Out


def expand(Lold, Minfrq, Maxfrq, Cl, Gl, Wpb):
# This function converts the chromosome to a new format where every bin
    # contains the list: [Amplitude, Phase, Frequency]

    Jfrq=(Maxfrq-Minfrq)/(Gl+1)
    # Jump frequency
    # This is the frequency gap between every two bins
    
    for frame in Lold:
        
        for i,bin in enumerate(frame):

            bin.append(i*Jfrq)
            # Adding the bin frequency to the bin
    # Now every bin contains [Amplitude, Phase, Frequency]

    return Lold


def frame_to_wav(F, Srate, SperF):
# Converts a frame from the chromosome to a frame in the wav format

    Wf=np.zeros(SperF, dtype=np.complex64)
    # Frame array
    # Create a complex array of the size of the samples per frame

    Cfrq=0
    # Initialize an Index variable at 0

    Count=0
    # Counts the number of times Cfrq has been incremented

    Jfrq= Srate/SperF


    for bin in F:

        while Cfrq<= bin[2]:
            Cfrq+= Jfrq
            Count+=1

        if Count>=SperF:
            break

        Wf[Count-1]+= (1-abs(Cfrq-bin[2])/Jfrq)* (bin[0]*np.cos(bin[1])+1j*bin[0]*np.sin(bin[1]))

        Wf[Count]+= (abs(Cfrq-bin[2])/Jfrq)* (bin[0]*np.cos(bin[1])+1j*bin[0]*np.sin(bin[1]))


    Wf=ifft(Wf)

    Wf=20000000*Wf

    Wf=np.array(Wf, dtype="int16")
    # Find the inverse fourier transform of this frame

    return Wf
