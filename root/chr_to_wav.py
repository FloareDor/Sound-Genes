import numpy as np
 
from scipy.io import wavfile

from scipy.fft import ifft

# My entire understanding of fft and hence ifft comes from this source:-
# http://howthefouriertransformworks.com/understanding-the-output-of-an-fft/

def decode(oldL, index, generation, minfrq, maxfrq, Cl, Gl, wavpbin, totalsamples, samplerate):
    
# Arguments explaination:-
# 1) oldL: This is the chromosome returned by DE. It does not contain the frequencies
#   nor corresponding to the amplitudes and phases
# 2) index: This is the index of the chromosome in each generation, used for file naming
# 3) generation: This is the generation number of the chromosome, used for file naming
# 4) minfrq: This is the minimum frequency
# 5) maxfrq: This is the maximum frequency
# 6) Cl: (Chromosome Length) This is the number of frames
# 7) Gl: (Gene Length) This is the number of bins in a frame
# 8) wavpbin: This is the number of waves in a bin that have the same amplitude, phase
# 9) totalsamples: This is the number of samples to be created in the entire audio file
# 10) samplerate: This is the number of samples per second


    L=[]
    # New chromosome with frequencies and bins

    jfrq=(maxfrq-minfrq)/(Gl+1)
    # Jump frequency
    # This is the frequency gap between every two bins

    for frame in oldL:
        L.append([])

        for i,bin in enumerate(frame):

            bin[0]=bin[0]*10000000
            # Scaling up the amplitude for fft

            bin.append(i*jfrq)
            # Adding the bin frequency to the bin

            x=bin

            # The below lines of code add the waves to the bin
            if wavpbin==1:
                L[-1].append(x)

            else:
                x[2]=x[2]-0.5*jfrq
                # Here 0.5 can be alterred to create waves that are more localised around
                # a specific bin

                for k in range(wavpbin):
                    incr=jfrq/(wavpbin-1)

                    if x[2]+k*incr<0:
                        L[-1].append([x[0],x[1], 0])
                    else:
                        L[-1].append([x[0],x[1],x[2]+k*incr])
    
    # Now every bin contains [Amplitude, Phase, Frequency]

    timefr=totalsamples/(Cl*samplerate)
    # This is the length of a timeframe in seconds

    framelen=int(samplerate*timefr)
    # This is the length of a frame, in samples not seconds

    out=np.zeros(totalsamples, dtype=np.int16)
    # This is the array that will be finally converted to a wav

    precision=1
    # This is the precision of the round function
    # Rounding is required to fit any wave into the inverse fourier array
    #   (according to my understanding from the above link)

    nwavs=len(L[0])
    # This is the total number of waves in every frame, here a new variable is used to
    #   avoid edge cases

    for fno in range(Cl):
    # For every frame

        frame=np.zeros(framelen, dtype=np.complex64)
        # Create a complex array of the size of the frame in samples

        idx=0
        # Initialize an Index variable at 0

        for i in range(framelen):
        # For every sample point in frame

            curfrq=round(i*samplerate/framelen, precision)
            # Current Frequency
            # This is the current frequency that this point in the frame array
            #   corresponds to
            # Adding some complex number here will create a wave of this frequency

            if curfrq<minfrq:
                continue
            # Skip past frequencies below the minimum
        
            if curfrq>samplerate:
                break
            # The above code is correct. Changing samplerate to maxfrq completely
            #    changes the output and no longer gives the correct wave

            while curfrq>L[fno][idx][2]:
            # L[fno][idx][2] is the frequency of the idx bin of the fno frame in the
            #   chromosome

                idx=idx+1
                # Essentially a two pointer approach for traversing the frame array
                #   and the chromosome to find the frequencies in the frame array
                #   that approximate to the frequencies of the chromosome

                if idx>nwavs-1:
                    break
                # Edge case

            if idx>nwavs-1:
                break
            # Edge case

            if curfrq==round(L[fno][idx][2], precision):
            # If the frequency of this point in the array is approximately equal to
            #   the frequency of this point in the chromosome
            
                frame[i]=L[fno][idx][0]*np.cos(L[fno][idx][1])+1j*L[fno][idx][0]*np.sin(L[fno][idx][1])
                # Then create a wave of given amplitude and phase
                # Here real part is amplitude*cos(phase) and imaginary part is
                #   amplitude*sin(phase)

        frame=np.array(ifft(frame), dtype="int16")
        # Find the inverse fourier transform of this frame

        if fno==Cl-1:
        # Edge case if the song is not perfectly divisible by the frame length

            for i in range(framelen):
                if (fno*framelen+i)<totalsamples:
                    out[fno*framelen+i]=frame[i]

        else:

            for i in range(framelen):
                out[fno*framelen+i]=frame[i]
            # Add the frame to the out array
            # Here fno*framelen gives the time point of the start of this frame

    wavfile.write(f"audio_output/gen{generation}-{index}.wav", samplerate, out)
    # Write the wavfile