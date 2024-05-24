'''
MEAN SQUARE ERROR DIFFERENTIAL EVOLUTION (DE)

This code file is meant to implement Differential Evolution on a mean square error
    function.


To understand this whole code file, please understand that the whole code is split into
    two parts:

a) Code that implements differential evolution
b) Code that exists for multiprocessing purposes

There is one specific Multiprocessing function used, i.e. multiprocessing.Pool.map_async(),
    where the multiprocessing library is an inbuilt python library. Please read up on
    this function's documentation, but put simply:

result= map_async(func, inp, processes= P, chunksize= C)
> The above line calls the function 'func' as many times as the size of the list 'inp'.
> Each call of func has the argument func(inp[i]) where i is the i'th call.
> P is the number of cores the code should utilize (max is automatically set to pc max).
> C is the number of calls that go as a batch to a single core. If C= 3 for example, then the 
    data for inp[i], inp[i+1], and inp[i+2] (for example) will be sent to a core together. 
    This reduces the total data transfer overhead. However, if for example there is 1 batch
    of size C remaining, then the entire batch will be sent to a single core, which could
    be slower than splitting it to 3 seperate cores.
> 'result' is a generator type object, same as 'range()'. It cannot be iterated over,
    each element has to be retrieved using result.get() method.

    
Please feel free to use your editor to change all variable names for better understanding.
'''


import numpy as np

import random as rd
# This is used to for random number generation wherever neccessary

from copy import deepcopy
# This is a crucial library used to create perfect copies of chromosomes
# A perfect copy is a recursive copy. If you put A=B where A and B are lists, then
    # any change made to B will be in A also, because their pointers are now same.

import os   
# This library is used to create/check folders/directories
# creating output directories if they do not exist


### HYPERPARAMETERS

A= 10
# Amplitude
# This is the maximum amplitude or loudness
# This scale only matters for DE. You will have to change the 'chr_to_wav.py', for any
    # actual change to the loudness.

Cp= 0.9
# Crossover Probability
# This is the amount of crossover between the original vector and the mutant
    # vector used to create the trial vector

K=0.7
# Coefficient used to generate the mutant vector

Fr=[3]
# Max coefficient scale used to generate 'F' variable in DE to generate the mutant vector

Gc= 50
# Generation Cycle
# Hyperparameter used for cyclical variation in local mutation
# Must not be 0


### EVOLUTION PARAMETERS

Ps= 5
# Population Size

Gs= 11
# Generation Size

new_or_cont= 0
# Used to determine whether to initialize DE with a completely new population or to
    # continue from the last saved population which should be saved in the 'final_chromosomes'
    # folder.
# Value of '0' means use a new population. Value of '1' means use the old population.

Lt= 0
# Local Search Threshold
# Above this value, DE will incorporate local search as well

St= 6
# State Variable
# Before this generation, DE will operate on twice compressed chromosomes.
# Beyond it, DE will operate on only once compressed chromosomes.


### MULTIPROCESSING ARGUMENTS

Pc= 5
# Number of processes
# This is the number of cores this code should parallely run on

Ch= 1
# Chunk size
# This is the size of the batch of inputs to be sent to a core at a time.


### MUSIC PARAMETERS

Cp= 10
# Compression Parameter
# This is the number of bins to be compressed into one bin in after compression.

Cl= 1
# Chromosome Length
# Frames per audio sample
# This is the number of Time Frames in a single song
# Note, a frame is our word for a list that contains all frequency bins from 0 to Maxfrq.
    # Each frame is meant to represent that part of the song. For example is Cl=2, then
    # each chromosome will have 2 lists within it. Each list will be responsible for
    # containing the frequency information of half the song.


Gl= 16000*30
# Gene Length
# Bins per frame
# This is the number of frequency Bins in a single Time Frame
# Each bin is one list of the format [Amplitude, Phase].


Wpb= 1
# Waves per bin
# This is the number of pure sine waves in a single Bin.
# For example, if Wpb=1, then each bin will correspond uniquely to one frequency that
    # plays in the output song. If Wpb=3, then there will be 3 separate frequency
    # values that will be created with the same amplitude and phase that will play.
    # These can be something like 1 Hz, 1.2 Hz and 1.4 Hz. Each of the frequencies are
    # designed to be equidistant from one another and the frequency values created by
    # adjacent bins. All this can be changed in OLDER versions of 'chr_to_wav.py'.
    # Please refer to github for those.


Frl= 1
# Frame Length
# This is an old variable not in use in the current version of 'chr_to_wav.py'
# This is the time in seconds for which a frame lasts
# IMPORTANT: Frl*Cl=60 must hold if Frl<1.

Minfrq = 0
# Minimum Frequency

Maxfrq = 8000
# Maximum Frequency

Srate=round(16000/Frl)
# Sampling Rate 
# This is the number of "Samples" in the song. Simply put, a sample is a single point
    # in an Magnitude-Time graph. If you put a chain of them together, we get sound.
# Please note that the Srate is innately tied to Gl, and conversion from chromosomes to
    # the wav format. I have made the code as unbreakable as possible but using random
    # values of Srate without changing 'chr_to_wav.py' will break something.
# The provided formula makes it so that in every frame the song is sampled 16000 
    # times.


# HERE ONWARDS PLEASE START READING FROM MAIN, IN ORDER OF THE CALLS

### FITNESS AND VALIDATION

# The below function was written by Ravi, hence the different style from mine.
## This fitnessFunction IS NOT IN USE here. It simply exists to create chromosomes.
def fitnessFunction(Inp):
    """
    Calculate fitness value for a given chromosome.
    
    Args:
        chromosome (list): The chromosome to calculate fitness for.
        index (int): Population number of the chromosome.
        generation (int): Generation number.
        rasaNumber (int, optional): The rasa number to consider (default is 1 which represents Karuna).

    Returns:
        float: The fitness value.
    """

    from chr_to_wav import decode
    # This is the function that converts chromosomes to wav files into the 'audio_output' folder

    chromosome=Inp[0]
    index=Inp[1]
    generation=Inp[2]
    rasaNumber=1
    # Note how every input value must be unpacked from the 'Inp' list. This is for parallelisation.

    rasas = ['Karuna', 'Shanta', 'Shringar', 'Veera']
    chromosome_copy = deepcopy(chromosome).tolist()
    # The above copy must be created to preserve the original

    for j in range(1,Cl):
        for k in range(Gl):
            chromosome_copy[j][k].append(chromosome_copy[0][k][1])
    # The way chromosome was designed is that only the first frame has phases, the rest use the same phases as the first.
    # However the converter requires a phase for each Amplitude, hence the above loops.

    decode(chromosome_copy, index=index, generation=generation, Minfrq=Minfrq, Maxfrq=Maxfrq, Cl=Cl, Gl=Gl, Wpb=Wpb, TS=Srate*60, Srate=Srate)
    # Create the wav formats

    return


## This is the Q value, sum of difference between aplitudes of adjacent bins
def Q(L):
    
    sum= 0

    for j in range(Cl):
        for k in range(L.shape[1]-1):

            sum+= abs(L[j][k][0]-L[j][k+1][0])

        if j!=Cl-1:
            sum+=abs(L[j][0][0]-L[j+1][k-1][0])

    return sum/2000

## This is the Mean Square Error Function
def ff(Inp):

    L=Inp[0]
    Gn=Inp[2]
    # Recieve inputs

    if Gn< St:
        Test= np.load(f'./initial_chromosomes/L2000cc.npz')
        Test=Test['arr_0']
        # If the chromosomes are double compressed, then use the doubly compressed target chromosme
    else:
        Test= np.load(f'./initial_chromosomes/L2000c.npz')
        Test=Test['arr_0']
        # If the chromosomes are singly compressed, then use the singly compressed target chromosme

    if Gn>=Gs:
        Test= np.load(f'./initial_chromosomes/L2000.npz')
        Test=Test['arr_0']
        # If they are not compressed at all, use this

    # Return the MSE of the chromosome with the Target and also return the Q values. The 0 is buffer value.
    return [np.mean(np.square(Test-L)), 0, Q(L)]


### ASSISTING FUNCTIONS

def chrm(new_or_cont, i):
# Random Chromosome Generator
# Creates a chromosome within the permissible boundary values

    # If we want a completely new population
    if new_or_cont==0:

        # The below statement is a recent addition, which allows completely random chromosomes
            # to be created. For example if we set i<10 instead of i<0, then the first 10
            # chromosomes will be created randomly
        if i<0:
            
            loaded = np.load(f'./initial_chromosomes/L0.npz')
            L=loaded['arr_0']
            # Load the chromosome from the initial chromosomes file which contains the rasas
                # in chromosome form

            for j in range(Cl):
                for k in range(Gl):
                    L[j][k][0]= rd.uniform(0,A)
                    L[j][k][1]= rd.uniform(0,6.282)

            # Overwrite the saved chromosome with completely random values within the bounds
            # This is faster than creating the chromosome structure from scratch using lists

            # Return the randomised chromosome
            return L

        # Else
        Rasas=[
        rd.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        rd.choice([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        rd.choice([20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
        rd.choice([30, 31, 32, 33, 34, 35, 36, 37, 38, 39]),
        ]
        # Choose a random element from each of the lists. This will be some integer.
        # Now use the 4 songs selected to generate an interpolated random mixture


        probabilities= [np.random.uniform(0.25, 0.75) for z in range(4)]
        probabilities= [z/sum(probabilities) for z in probabilities]
        # Create some weightage for each of the 4 rasas normalised to add to 1.

        for z, rasa in enumerate(Rasas):
            loaded = np.load(f'./initial_chromosomes/L{rasa}.npz')
            T=loaded['arr_0']

            if z==0:
                L=probabilities[z]*T
            else:
                L+=probabilities[z]*T

            # For each integer in 'Rasas', load it and take the weighted sum of them as the output chromosome. 

        L=np.array(L, dtype= "float32")
        # This step makes the format uniform after the previous operations.
    
        return L

    # If we want to continue with the chromosomes produced in the previous save state
    else:

        loaded = np.load(f'./final_chromosomes/L{i}.npz')
        L=loaded['arr_0']

        L=np.array(L, dtype= "float32")

        # Just load all the chromosomes from the folder and return

        return L


def encoder(L, compression, i):
# Takes a chromosome L and returns a compressed version where every k bins are averaged
    # to one bin. k= 'compression' in input.

    T=[]
    
    for j in range(Cl):
        T.append([])

        for k in range(0, L.shape[1], compression):
            T[j].append(np.average(L[0][k:k+10], axis=0))

    T=np.array(T)
    # Create the compressed chromosome

    # New change, to avoid losing randomness due to compression
    # For example if i<10, then the first 10 chromosomes will be completely randomised.
    # This is different from randomising at the uncompressed level because after compression
        # everything would end up near to the mean
    if i<2*Ps/3:

        for j in range(Cl):
            for k in range(T.shape[1]):
                T[j][k][0]= rd.uniform(0,A)
                T[j][k][1]= rd.uniform(0,6.282)

    return T

def distribution(Initial, compression, Final):
    '''
    Takes a chromosome's uncompressed (Initial) and compressed(Final) values and returns the distribution of the chromosome.
    Eg. If 10 bins are compressed to 1 bin using an average, then the distribution vector
    will be a 10 size vector where each element is the difference between the original
    chromosome and the mean.
    '''

    T= []
    
    for i in range(Cl):
        T.append([])

        for j in range(0, Final.shape[1]):
            for k in range(compression):
                T[i].append(Initial[i][j*compression+k]-Final[i][j])

    T=np.array(T)
    # This simply applies the formula for finding the distribution.

    return T

def decoder(L, compression, Distribution):
    '''
    Takes a chromosome L and returns a decompressed version where every bin is expanded to
    k bins using the distribution.
    '''
    T=[]
    
    for i in range(Cl):
        T.append([])

        for j in range(0, L.shape[1]):
            for k in range(compression):

                T[i].append(L[i][j]+Distribution[i][j*compression+k])

    T=np.array(T)

    return T

def encodeRoutine(callno):

    if callno==1:
        Test= np.load(f'./initial_chromosomes/L2000.npz')
        Test=Test['arr_0']
    else:
        Test= np.load(f'./initial_chromosomes/L2000c.npz')
        Test=Test['arr_0']

    # callno is used to understand at what level the compression is currently at


    Testcompressed= encoder(Test, compression=Cp, i=2000)
    # Compress the Target song

    if callno==1:
        np.savez_compressed(f"./initial_chromosomes/L2000c.npz", Testcompressed)
    else:
        np.savez_compressed(f"./initial_chromosomes/L2000cc.npz", Testcompressed)
    # Save the compressed version incase it wasn't already saved

    Distribution= distribution(Test, Cp, Testcompressed)
    # Find the distributions for decompression


    for i in range(Ps):
        Pop[i]= encoder(Pop[i], compression= Cp, i=i)
    # Compress the rest of the population

    return Distribution

def decodeRoutine(Distribution):

    for i in range(Ps):
        Pop[i]= decoder(Pop[i], compression= Cp, Distribution= Distribution)
    # Simply use the Distribution to decode the population

    return


def popinit(new_or_cont):
# Population Initiator
# Creates an initial population pool

    from multiprocessing import Pool
    # For parallelization

    for i in range(Ps):
        Pop[i]=chrm(new_or_cont, i)
    # For each population member, create a chromosome using the 'chrm()' function and
        # store it in 'Pop' the population dictionary
    # Every element in Pop is a Chromosome indexed from an integer from 0 to Ps

    Distributionc= encodeRoutine(1)
    Distributioncc= encodeRoutine(2)
    # These are distributions of chromosome created for the sake of decompressing
    # It is only the distribution of the Target song (That is used in MSE)
    # All other chromosomes will also be decompressed using the same distribution

        
    Inp=[[Pop[i], i, 0] for i in range(Ps)]
    # Packing inputs for parallel calls

    # Start the multiprocessing
    with Pool(processes= Pc) as pool:

        result = pool.map_async(ff, Inp, chunksize= Ch)
        # Send the parallel calls

        for Out in result.get():
            Fitness.append(Out[0])
            SFitness.append(Out[1])
            QFitness.append(Out[2])
        # Unpack the return values

    # Return the distribution vectors.
    return [Distributionc, Distributioncc]


def fittest():
# Fittest Population Member Evaluator
# Finds the Population Member with the Greatest Fitness and that Fitness
# Redundant function, the same thing can be done in 1 line (I was new to python)

    Bf=min(Fitness)
    Fiti=Fitness.index(Bf)

    return Fiti, Bf


### CORE FUNCTIONS

def poprun(Inp):


    i= Inp[0]
    # 1) This is the index of the chromosome this call must work on
    Gn= Inp[1][0]
    # 2) Generation Number
        # This is used in the fitness function
    Pop= Inp[2]
    # 3) This is the population dictionary that contains all chromosomes
    Fiti= Inp[3][i]
    # 4) This is the fitness of the current chromosome
       # It is not evaluated natively to reduce number of fitness function evaluations
    X= Inp[4][0][0]
    # 5) This is the fittest chromosome from the previous generation
       # It is used for Global search
   
    Local_Copy= deepcopy(Pop[i])

    for f in range(1, 20):
    # This is the maximum number of times the chromosome should be revaluated if it is out of bounds

        Mut=deepcopy(Local_Copy)
        # Mutant Vector
        # This is the Mutant Vector of population member index i
        # Initiate it as a random chromosome

        Tri= deepcopy(Local_Copy)
        # Trial Vector
        # Tri is the Trial Vector of the population member L
        # Initiate it to be the Mutant Vector

        if f<4:
            F=rd.uniform(-Fr[0]/f, Fr[0]/f)
        else:
            F=rd.uniform(-1, 1)
        # Coefficient used to generate the mutant vector
        # Gradually reduce the boundaries but clamp it to a minimum of (-1, 1)

        while(1):
            
            y,z= rd.sample(range(Ps),2)

            if(y!=i and z!=i and y!=X and z!=X):
                break
        # The above are used to choose population vectors to mutate the original 
            # vector with
        

        Mut+= K*(Pop[X]-Pop[i])+F*(Pop[y]-Pop[z])
        # There's a reason it is done in the above way!!
        # Use Numpy constructs. Doing it using loops is extremely slow.

        # Mut is now the Mutant Vector of population member L


        # Now we must create the Trial vector
        Temp= 0 # This variable will be 1 if the Trial vector goes out of bounds

        for j in range(Cl):
            for k in range(Local_Copy.shape[1]):

                if j==0:
                    for l in range(2):
                        if (rd.uniform(0,1)< Cp):
                            Tri[j][k][l]= Mut[j][k][l]

                else:
                    if (rd.uniform(0,1)< Cp):
                        Tri[j][k][0]= Mut[j][k][0]

                    # Here as the Tri is the same as Mut initially, elements of Tri
                        # are restored to the value of L with a probability of 1-Cp

                if (Tri[j][k][0]> A+A/10 or Tri[j][k][0]<0-A/10):
                    Temp= 1
                
                elif (Tri[j][k][0]> A):
                    Tri[j][k][0]= A

                elif (Tri[j][k][0]< 0):
                    Tri[j][k][0]= 0
                    
                if j==0:
                    if (Tri[j][k][1]>6.282+1 or Tri[j][k][1]<0-1):
                        Temp= 1

                    elif(Tri[j][k][1]>6.282):
                        Tri[j][k][1]= 6.282

                    elif(Tri[j][k][1]< 0):
                        Tri[j][k][1]= 0

                if Temp==1:
                    break

            if Temp==1:
                break
        
        if Temp==0:
            break

        # Tri is now the Trial Vector of the population member L
        # If none of the constraints are violated (Temp==0) then break, else revaluate
            # from the mutation vector


    # If after all the above code the trial vector is still out of bounds, then
        # just make the out of bounds values the same as the values of the vector's parent
    if Temp!=0:
        for j in range(Cl):
            for k in range(Local_Copy.shape[1]):

                if (Tri[j][k][0]> A or Tri[j][k][0]<0):

                    Tri[j][k][0]=Pop[i][j][k][0]
                    
                if j==0:
                    if (Tri[j][k][1]>6.282 or Tri[j][k][1]<0):

                        Tri[j][k][1]=Pop[i][j][k][1]
    # If a component of the Trial Vector is Violating a constraint, replace
        # that component with that of the population member


    Temp=ff([Tri, i, Gn])
    # Temp=ff(Tri)
    # Temp is used here to reduce the number of fitness function calls

    if(Temp[0]<=Fiti):
        return [Tri, Temp]
        # If successful, return the child chromosome as well as it's fitness

    else:
        return 0
        # If failure, return 0
 

def loccro(Inp):
# Basic implementation of Local Crossover

    i=Inp[0]

    if i==0:
        return 0

    Gn=Inp[1][0]
    Pop=Inp[2]
    # Fiti requires the arguments from Inp[4]
    Ind1=Inp[4][i] # This is the index of the current chromosome, not 'i'
    Ind2=Inp[4][i-1] # This is the index of the chromosome with fitness better than chromosme with index Ind1

    Fiti=Inp[3][Ind2]
    # Unpack inputs


    Cro=deepcopy(Pop[Ind1])
    # Create local copy

    R=rd.uniform(0,1)

    Cro+= R*(Pop[Ind2]-Cro)
    # This is the core equation


    Temp=ff([Cro, i, Gn])

    if(Temp[0]<=Fiti):
        return [Cro, Temp]
        # If successful, return the child chromosome as well as it's fitness

    else:
        return 0
        # If failure, return 0


def locmut(Inp):
# Basic implementation of Local Mutation

    i=Inp[0]
    Gn=Inp[1][0]
    Pop=Inp[2]
    Fiti=Inp[3][i]
    Min=Inp[4][0][0]
    Max=Inp[4][1][0]
    Pm=Inp[4][2][i] # This is ready for multiplication, not done n-i way.
    Cyc=Inp[4][3][0]


    Mut=deepcopy(Pop[i])


    for j in range(Cl):
        for k in range(Mut.shape[1]):
            
            R=rd.uniform(0,1)

            if j==0:
                for l in range(2):
                
                    x=rd.randint(0,1)

                    if x==0:
                        Mut[j][k][l]+= R*Cyc*Pm*(Max[j][k][l]-Mut[j][k][l])
                    else:
                        Mut[j][k][l]+= R*Cyc*Pm*(-Min[j][k][l]+Mut[j][k][l])
            
            else:
                x=rd.randint(0,1)

                if x==0:
                    Mut[j][k][0]+= R*Cyc*Pm*(Max[j][k][0]-Mut[j][k][0])
                else:
                    Mut[j][k][0]+= R*Cyc*Pm*(-Min[j][k][0]+Mut[j][k][0])


    Temp=ff([Mut, i, Gn])

    if(Temp[0]<=Fiti):
        return [Mut, Temp]
        # If successful, return the child chromosome as well as it's fitness

    else:
        return 0
        # If failure, return 0


def localsearch(Inp):
# Hub for calling all the different DE functions and updating the list values
# Each call corresponds to one 'i' value, hence to one chromosome

    Inp1= Inp[0]
    Inp2= Inp[1]
    Inp3= Inp[2]
    Inp4= Inp[3]
    # Unpack all the function inputs we will use


    i=Inp1[0]
    # This chromosome
    Pop=Inp1[2]
    Fitness=Inp1[3]
    Gn=Inp1[1][0]
    Indsort=Inp2[4]
    SFitness=Inp4[0]
    QFitness=Inp4[1]
    # The above are the values that we will be using in this function


    # If Generation Number is greater than Local search Threshold, perform Local Search,
        # else only perform regular DE
    if Gn>Lt:

        Out= loccro(Inp2)
        # Call local crossover

        if Out!=0:

            Pop[Indsort[i]]=Out[0]

            Fitness[Indsort[i]]=Out[1][0]
            SFitness[Indsort[i]]=Out[1][1]
            QFitness[Indsort[i]]=Out[1][2]
        # Unpack outputs


        Out= locmut(Inp3)
        # Call local mutation

        if Out!=0:

            Pop[i]=Out[0]

            Fitness[i]=Out[1][0]
            SFitness[i]=Out[1][1]
            QFitness[i]=Out[1][2]


    Out= poprun(Inp1)
    # Call global search DE

    if Out!=0:

        Pop[i]=Out[0]

        Fitness[i]=Out[1][0]
        SFitness[i]=Out[1][1]
        QFitness[i]=Out[1][2]

    return



def main():

    from multiprocessing import Pool, Manager
    # This is a crucial library used to implement parallel processing in the algorithm

    import time
    # This is simply used to calculate the total execution time of the code

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    # This is used for plotting purposes


    Start= time.time()
    # Just to time the code


    manager= Manager()
    # The manager creates shared memory datatypes. 
    # These datatypes can be accessed by any processor at any time as long as the pointer
        # is passed in it's input list.
    # However, accessing variables from such datatypes is very time consuming (more than
        # 10 times slower for chromosomes), hence whenever we want to repeatedly use some
        # data, we should create a local copy of it using deepcopy, and then use a copy
        # of that copy for mutations, etc.

    global Pop
    Pop= manager.dict()
    # Population Dictionary
    # This is a dictionary mapping unique integers to unique Chromosomes
    # Initiate it as an empty dictionary
    # This will be in the shared memory

    global Fitness
    Fitness= manager.list()
    # Fitness List
    # This is a list that stores the fitness of every population member at any given time
    # Initiate it as an empty list

    global Bfit
    Bfit=[]
    # Best Fitness List
    # This is a list of the best fitness in every generation

    Afit=[]
    # Average Fitness List
    # This is a list of the average fitness in every generation for plotting purposes

    global SFitness
    SFitness= manager.list()
    # SOM Fitness of all population members
    # SOM Fitness and MSE Fitness have been differentiated here


    Sval=[]
    # SOM Fitness of fittest population member
    Savg=[]
    # Average SOM Fitness of population

    global QFitness
    QFitness= manager.list()
    # Q values of all population members

    Qval=[]
    # Q values of fittest population member
    Qavg=[]
    # Average Q value of population


    Out=popinit(new_or_cont)
    # Initiate the Population. Please skip to the function 'popinit()' now
    Distributionc= Out[0]
    Distributioncc= Out[1]
    # Get the Distributions for decompression
    # Distributionc is for decompressing from singly compressed to uncompressed.
    # Distributioncc is for decompressing from doubly compressed to singly compressed.
    

    Gn=0
    # Generation Number
    # Used to determine what generation the population is currently at.


    # Multiple lines for packing Input values for multiprocessing.
    # Remember that everything must be put into a list so that the pointers from
        # Inp to this item is maintained, even if the item within the list is changed.
    # This is done to prevent having to create the list over and over everytime a variable's
        # value is changed.

    Best=[fittest()]
    Gen=[Gn,Gc]
    Inp1=[[i, Gen, Pop, Fitness, Best] for i in range(0, Ps)]

    Ind=[i for i in range(0, Ps)]
    Indsort=deepcopy(Ind)
    Inp2=[[i, Gen, Pop, Fitness, Indsort] for i in range(0, Ps)]

    Min=[deepcopy(Pop[0])]
    Max=[deepcopy(Pop[0])]
    Pm=[0]*Ps
    Cyc=[0]
    Inp3=[[i, Gen, Pop, Fitness, [Min, Max, Pm, Cyc]] for i in range(0, Ps)]

    Inp4=[[SFitness, QFitness] for i in range(Ps)]


    Inp=[[Inp1[i], Inp2[i], Inp3[i], Inp4[i]] for i in range(Ps)]

    # 'Inp' is the input to the localsearch function 


    # While Generation Number is less than Generation Size (Max no of generations)
    while Gn<Gs:

        Gn=Gn+1
        print("Gen: ", Gn)

        # Such statements are scattered throughout. They are used for updating the values
            # in the 'Inp' list created above.
        Gen[0]= Gn
        Cyc[0]= np.exp(-2*(Gn%Gc)/Gc)* 1/(1+np.exp(-1*((Gc/2)-(Gn%Gc))))


        if Gn==St:
            decodeRoutine(Distributioncc)
        if Gn== Gs:
            decodeRoutine(Distributionc)
        # When to decompress the chromosomes


        with Pool(processes= Pc) as pool:
        # Create an instance of the pool process and call it "pool"

            if Gn>Lt:
            # If Local Search Threshold is surpassed, perform Local Search as well

                Junk,Temp=zip(*sorted(zip(Fitness, Ind)))
                # Sort all the chromosomes by their fitness and maintain their index values
                # Sorted in descending order as lower fitness is better

                for i in range(0, Ps):
                    Indsort[i]=Temp[i]

                    Pm[Temp[i]]=i/Ps
                # Pm is used in Local Search, and Indsort will store all the indices of
                    # the chromosomes in descending order of their fitness


                # Every 10'th generation, update the Max and Min vectors used in local search.
                # Only every 10'th generation because it is very time consuming
                if Gn%10==1 or Gn==St or Gn==Gs-1:

                    Popcopy=deepcopy(Pop)
                    # DO NOT WORK WITH THE ORIGINAL
                    # This is because it is a shared memory datatype, extremely expensive to access repeatedly

                    if Gn==St or Gn==Gs:
                        Min[0]=deepcopy(Pop[0])
                        Max[0]=deepcopy(Pop[0])

                    for i in range(Ps):
                        for j in range(Cl):
                            for k in range(Popcopy[0].shape[1]):

                                if i==0:
                                    if j==0:
                                        for l in range(2):
                                            Min[0][j][k][l]= Popcopy[i][j][k][l]
                                            Max[0][j][k][l]= Popcopy[i][j][k][l]

                                    else:
                                        Min[0][j][k][0]= Popcopy[i][j][k][0]                               
                                        Max[0][j][k][0]= Popcopy[i][j][k][0]


                                if j==0:
                                    for l in range(2):
                                        if (Popcopy[i][j][k][l]<Min[0][j][k][l]):
                                            Min[0][j][k][l]= Popcopy[i][j][k][l]
                        
                                        elif(Popcopy[i][j][k][l]>Max[0][j][k][l]):
                                            Max[0][j][k][l]= Popcopy[i][j][k][l]

                                else:
                                    if (Popcopy[i][j][k][0]<Min[0][j][k][0]):
                                        Min[0][j][k][0]= Popcopy[i][j][k][0]
                        
                                    elif(Popcopy[i][j][k][0]>Max[0][j][k][0]):
                                        Max[0][j][k][0]= Popcopy[i][j][k][0]


            result = pool.map_async(localsearch, Inp, chunksize= Ch)
            # For every population member, initiate local search with parallel processing
            # The outputs will be stored in the generator "result"

            for _ in result.get():
            # ".get()" is used to extract outputs from a generator
                
                pass


        # Store the values in their respective lists
        Best[0]=fittest()

        Afit.append(sum(Fitness)/float(Ps))
        Bfit.append(Best[0][1])

        Sval.append(SFitness[Best[0][0]])
        Savg.append(sum(SFitness)/Ps)

        Qval.append(QFitness[Best[0][0]])
        Qavg.append(sum(QFitness)/Ps)

        print("Best= ", Best[0], Sval[-1], Qval[-1])


        # Every 10th generation create graphs, store values, create copies etc.
        if(Gn%10==0):
            plt.plot(range(0,len(Afit)), Afit, label="Avg Fitness")
            plt.plot(range(0,len(Bfit)), Bfit, label="Best Fitness")
            plt.xlabel("Number of Generations")
            plt.ylabel("Fitness")
            plt.legend(loc="upper right")
            plt.savefig(f"graphs/Graph.png")
            plt.close()
        

            # This file is extremely important and has to be used for everything
            with open("Values_LSDE.txt","w") as f:

                for i in range(len(Qval)):
                    print(Afit[i], file=f, end=",")
                    print(Bfit[i], file=f, end=",")
                    print(Sval[i], file=f, end=",")
                    print(Savg[i], file=f, end=",")
                    print(Qval[i], file=f, end=",")
                    print(Qavg[i], file=f, end="\n")


            # Here store all the final chromosome values only every 100th generation
            # This is time taking because it requires decompression for storing
            if (Gn%100==0):
                if Gn<St:
                    decodeRoutine(Distributioncc)
                    decodeRoutine(Distributionc)
                else:
                    decodeRoutine(Distributionc)

                for i in range(Ps):

                    Array= np.array(Pop[i], dtype="float32")
                    np.savez_compressed(f"./final_chromosomes/L{i}.npz", Array)

                    if i==Best[0][0]:
                        fitnessFunction([Pop[Best[0][0]], 1000, Gn])

                if Gn<St:
                    encodeRoutine(1)
                    encodeRoutine(2)
                else:
                    encodeRoutine(2)
                # Recompress them before proceeding
                


    for i in range(Ps):
        Array= np.array(Pop[i], dtype="float32")
        np.savez_compressed(f"./final_chromosomes/L{i}.npz", Array)
    # Store all the Final Chromosomes incase we want to use them

    End= time.time()

    print(Best, End-Start)


if __name__ == '__main__':

    # Create all the folders incase they are deleted    

    audioOutputPath = 'audio_output'
    featuresOutputPath = 'features_output'
    graphsPath = 'graphs'
    finalChromosomesPath = 'final_chromosomes'
    if not os.path.exists(audioOutputPath):
        os.makedirs(audioOutputPath)
    else:
        pass
    if not os.path.exists(featuresOutputPath):
        os.makedirs(featuresOutputPath)
    else:
        pass
    if not os.path.exists(graphsPath):
        os.makedirs(graphsPath)
    else:
        pass
    if not os.path.exists(finalChromosomesPath):

        os.makedirs(finalChromosomesPath)
    else:
        pass

    # Start the code
    main()