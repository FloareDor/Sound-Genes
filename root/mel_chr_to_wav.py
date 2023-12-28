### Mel frequency integration

import numpy as np
 
from scipy.io import wavfile

from scipy.fft import ifft

# My entire understanding of fft and hence ifft comes from this source:-
# http://howthefouriertransformworks.com/understanding-the-output-of-an-fft/

# Max frequency of set-> Number of bins, Number of waves per bin
Fsets={
    399:(57,7),
    799:(40,10),
    1202:(31,13),
    1602:(25,16),
    2001:(21,19),
    2397:(18,22),
    2797:(16,25),
    3189:(14,28),
    3592:(13,31),
    4000:(12,34),
    4407:(11,37),
    4807:(10,40),
    5203:(9,44),
    5599:(9,44),
    5999:(8,50),
    6399:(8,50),
    6798:(7,57),
    7197:(7,57),
    7596:(7,57),
    7998:(6,67)
}


def decode(Lold, index, generation, Minfrq, Maxfrq, Cl, Gl, TS, Srate):

    Minfrq=0
    Maxfrq=8000
    Cl=120
    Gl=329
    TS=Srate*60
    Srate=32000

    
# Arguments explaination:-
# 1) Lold: This is the chromosome returned by DE. It does not contain the frequencies
#   nor corresponding to the amplitudes and phases
# 2) index: This is the index of the chromosome in each generation, used for file naming
# 3) generation: This is the generation number of the chromosome, used for file naming
# 4) Minfrq: This is the minimum frequency
# 5) Maxfrq: This is the maximum frequency
# 6) Cl: (Chromosome Length) This is the number of frames
# 7) Gl: (Gene Length) This is the number of bins in a frame
# 8) TS: (Total Samples) This is the number of samples in the entire created audio file
# 9) Srate: (Sample Rate) This is the number of samples per second


    L=expand(Lold=Lold, Minfrq=Minfrq)
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
        # return Wf

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


def expand(Lold, Minfrq):
# This function converts the chromosome to a new format where every bin
    # contains the list: [Amplitude, Phase, Frequency]

    L=[]
    # New chromosome with frequencies and bins


    for frame in Lold:
        L.append([])

        Cfrq=0

        idx=0

        for Frq in Fsets:
            
            Wpb=Fsets[Frq][1]

            Jfrq=400/(Fsets[Frq][0]+1)

            Cfrq+=Jfrq/2


            for _ in range(Fsets[Frq][0]):

                while Cfrq<Minfrq:
                    Cfrq+=Jfrq

                frame[idx][0]=frame[idx][0]*1500000/Wpb
                # Scaling up the amplitude for fft
                frame[idx].append(Cfrq)
                # Adding the bin frequency to the bin

                # The below lines of code add the waves to the bin

                Cfrq=Cfrq+Jfrq

                if Cfrq>Frq:
                    Cfrq=Frq


                if Wpb==1:
                    L[-1].append(frame[idx])

                else:
                    frame[idx][2]=frame[idx][2]-0.45*Jfrq
                    # Here 0.5 can be alterred to create waves that are more localised around
                    # a specific bin
                
                    incr=0.9*Jfrq/(Wpb-1)
                    for k in range(Wpb):
                        L[-1].append([frame[idx][0],frame[idx][1],frame[idx][2]+k*incr])

                idx+=1

    # Now every bin contains [Amplitude, Phase, Frequency]

    return L


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


    Ctprv=0
    Ctrep=1


    for bin in F:
        Ctprv=Count

        while Cfrq<= bin[2]:
            Cfrq+= Jfrq
            Count+= 1

        if Count==Ctprv:
            Ctrep+=1
        else:
            Wf[Ctprv-1]=Wf[Ctprv]/Ctrep
            Wf[Ctprv]=Wf[Ctprv]/Ctrep

            Ctrep=1

        Wf[Count-1]+= (abs(Cfrq-bin[2])/Jfrq)* (bin[0]*np.cos(bin[1])+1j*bin[0]*np.sin(bin[1]))
        
        Wf[Count]+= (1-abs(Cfrq-bin[2])/Jfrq)* (bin[0]*np.cos(bin[1])+1j*bin[0]*np.sin(bin[1]))


    Wf=np.array(ifft(Wf), dtype="int16")
    # Find the inverse fourier transform of this frame
    return Wf