import numpy as np

import random as rd
# This is used to for random number generation wherever neccessary

from copy import deepcopy
# This is a crucial library used to create perfect copies of chromosomes


### HYPERPARAMETERS

A= 200000000
# Amplitude
# This is the maximum amplitude or loudness

Cp= 0.95
# Crossover Probability
# This is the amount of crossover between the original vector and the mutant
    # vector used to create the trial vector

K=0.5
# Coefficient used to generate the mutant vector

Fr=[2]
# Coefficient used to generate the mutant vector


### EVOLUTION PARAMETERS

Ps= 5
# Population Size

Gs= 2
# Generation Size

new_or_cont= 0


### MULTIPROCESSING ARGUMENTS

Pc= 5
# Number of processes
# This is the number of cores this code should parallely run on

Ch= 1
# Chunk size
# This is the number of elements each parallel run should process before returning


### MUSIC PARAMETERS

Cl= 1
# Chromosome Length
# Frames per audio sample
# This is the number of Time Frames in a single song

Gl= 16000*30
# Gene Length
# Bins per frame
# This is the number of frequency Bins in a single Time Frame

Wpb= 1
# Waves per bin
# This is the number of waves in a single Bin


Frl= 1
# Frame Length
# This is the time in seconds for which a frame lasts
# IMPORTANT: Frlen*Cl=60 must always hold

Minfrq = 0
# minimum frequency per frame

Maxfrq = 8000
# maximum frequency per frame

Srate=round(16000/Frl)
# Sampling Rate 
# The provided formula makes it so that in every frame the song is sampled 16000 
    # times. (In my opinion this does not work and it should just be 16000)



### FITNESS AND VALIDATION

from som.som_class import SOM

som = SOM()
# Initialization of Self-Organizing-Map

# Use the below code if there are multiple ffs
ff1i=1.8 # Average value of ff1
ff2i=1.4*100000000 # Average value of ff1
s1=1
s2=ff1i/ff2i
w1=0.85 # Weight of ff1
w2=0.15 # Weight of ff2


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
    # This is the function that converts chromosomes to wav files.

    from driver import computeFitnessValues, compute_SOM_DOB
    # This is the fitness function that utilizes 5 features from set A and another 5 from set B


    chromosome=Inp[0]
    index=Inp[1]
    generation=Inp[2]
    # generation=Inp[2][0]
    rasaNumber=1

    rasas = ['Karuna', 'Shanta', 'Shringar', 'Veera']
    chromosome_copy = deepcopy(chromosome).tolist()

    for j in range(1,Cl):
        for k in range(Gl):
            chromosome_copy[j][k].append(chromosome_copy[0][k][1])

    decode(chromosome_copy, index=index, generation=generation, Minfrq=Minfrq, Maxfrq=Maxfrq, Cl=Cl, Gl=Gl, Wpb=Wpb, TS=Srate*60, Srate=Srate)

    rasaDob = compute_SOM_DOB(SOM=som,audioFile=f"gen{generation}-{index}.wav", generation=generation, populationNumber=index)
    fitnessValue = rasaDob[rasaNumber]
    print("rasaDOB: ",rasaDob)

    # values = dict(computeFitnessValues(rasaNumber=rasaNumber, audioFile=f"gen{generation}-{index}.wav", generation=generation, populationNumber=index))
    # fitnessValue = float(values["fitnessValues"][rasas[rasaNumber-1]]["weightedSum"])

    # return fitnessValue**2

    return w1*s1*(fitnessValue**2)+w2*s2*Q(chromosome)


def ff(L):

    import numpy as np

    loaded= np.load(f'./initial_chromosomes/L0.npz')

    Test=loaded['arr_0']

    La= np.array(L, dtype='float32')

    return np.mean(np.square(Test-La))


def Q(L):
    
    sum= 0

    for j in range(Cl):
        for k in range(Gl-1):

            sum+= abs(L[j][k][0]-L[j][k+1][0])

        if j!=Cl-1:
            sum+=abs(L[j][0][0]-L[j+1][k-1][0])

    return sum/2000


### ASSISTING FUNCTIONS

def chrm(new_or_cont, i):
# Random Chromosome Generator
# Creates a random chromosome with the permissible boundary values
    if new_or_cont==0:

        Rasas=[
        rd.choice([0, 1, 2, 4]),
        rd.choice([19, 23]),
        rd.choice([33, 36, 37, 38, 39]),
        rd.choice([47, 48, 49, 50, 51, 59])
        ]

        probabilities= [np.random.uniform(0.25, 0.75) for z in range(4)]
        probabilities= [z/sum(probabilities) for z in probabilities]

        for z, rasa in enumerate(Rasas):
            loaded = np.load(f'./initial_chromosomes/L{rasa}.npz')
            T=loaded['arr_0']

            if z==0:
                L=probabilities[z]*T
            else:
                L+=probabilities[z]*T
    
        L=np.array(L, dtype= "float32")

        return L

    else:

        loaded = np.load(f'./final_chromosomes/L{i}.npz')
        L=loaded['arr_0']

        L=np.array(L, dtype= "float32")

        return L



def popinit(new_or_cont):
# Population Initiator
# Creates an initial population pool

    from multiprocessing import Pool

    for i in range(Ps):
        Pop[i]=chrm(new_or_cont, i)
        
    Inp=[[Pop[i], i, 0] for i in range(Ps)]
    # Inp=[Pop[i] for i in range(Ps)]

    with Pool(processes= Pc) as pool:

        result = pool.map_async(fitnessFunction, Inp, chunksize= Ch)

        for Out in result.get():
            Fitness.append(Out)

    # Every element in Pop is a Chromosome indexed from an integer from 0 to Ps

    return Pop


def fittest():
# Fittest Population Member Evaluator
# Finds the Population Member with the Greatest Fitness and that Fitness

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
   

    for _ in range(20):
    # This is the maximum number of times the chromosome should be revaluated if it is out of bounds

        Mut=deepcopy(Pop[i])
        # Mutant Vector
        # This is the Mutant Vector of population member index i
        # Initiate it as a random chromosome

        F=rd.uniform(-Fr[0], Fr[0])
        # Coefficient used to generate the mutant vector

        while(1):
            
            y,z= rd.sample(range(Ps),2)

            if(y!=i and z!=i and y!=X and z!=X):
                break
            
        # The above are used to choose population vectors to mutate the original 
            # vector with

        Temp= 0

        for j in range(Cl):
            for k in range(Gl):

                if j==0:
                    for l in range(2):
                        if (rd.uniform(0,1) < Cp):
                            Mut[j][k][l]+= K*(Pop[X][j][k][l]-Pop[i][j][k][l])+F*(Pop[y][j][k][l]-Pop[z][j][k][l])
                else:
                    if (rd.uniform(0,1) < Cp):
                        Mut[j][k][0]+= K*(Pop[X][j][k][0]-Pop[i][j][k][0])+F*(Pop[y][j][k][0]-Pop[z][j][k][0])

                    # The above is a direct formula used to create mutant vectors in DE

                if (Mut[j][k][0]> A+A/10 or Mut[j][k][0]<0-A/10):
                    Temp= 1
                
                elif (Mut[j][k][0]> A):
                    Mut[j][k][0]= A

                elif (Mut[j][k][0]< 0):
                    Mut[j][k][0]= 0
                    
                if j==0:
                    if (Mut[j][k][1]>6.282+1 or Mut[j][k][1]<0-1):
                        Temp= 1

                    elif(Mut[j][k][1]>6.282):
                        Mut[j][k][1]= 6.282

                    elif(Mut[j][k][1]< 0):
                        Mut[j][k][1]= 0

                if Temp==1:
                    break

            if Temp==1:
                break
        
        if Temp==0:
            break


    if Temp!=0:
        for j in range(Cl):
            for k in range(Gl):

                if (Mut[j][k][0]> A or Mut[j][k][0]<0):

                    Mut[j][k][0]=Pop[i][j][k][0]
                    
                if j==0:
                    if (Mut[j][k][1]>6.282 or Mut[j][k][1]<0):

                        Mut[j][k][1]=Pop[i][j][k][1]

    # If a component of the Trial Vector is Violating a constraint, replace
        # that component with that of the population member

    Temp=fitnessFunction([Mut, i, Gn])
    # Temp=ff(Mut)
    # Temp is used here to reduce the number of fitness function calls

    if(Temp<=Fiti):
        return [Mut, Temp]
        # If successful, return the child chromosome as well as it's fitness
        
    else:
        return 0
        # If failure, return 0
    

def main():

    from multiprocessing import Pool
    # This is a crucial library used to implement parallel processing in the algorithm

    import time
    # This is simply used to calculate the total execution time of the code

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    # This is used for plotting purposes

    import numpy as np
    # This is used for some calculations

    import os
    # This library is used to create/check folders/directories



    Start= time.time()

    global Pop
    Pop={}
    # Population Dictionary
    # This is a dictionary mapping integers to a unique Chromosome
    # Initiate it as an empty dictionary

    # FitnessSOM=[]
    # AfitSOM=[]
    # BfitSOM=[]
    # QvalSOM=[]

    global Fitness
    Fitness=[]
    # Fitness List
    # This is a list that stores the fitness of every population member at any given time
    # Initiate it as an empty list

    global Bfit
    Bfit=[]
    # Best Fitness List
    # This is a list of the best fitness in every generation for plotting purposes
    # Will be eliminated from the final code

    Afit=[]
    # Average Fitness List
    # This is a list of the average fitness in every generation for plotting purposes
    # Will be eliminated from the final code

    Qval=[]
    Qavg=[]
    # Q Values List
    # This is a list of the Q values in every generation for plotting purposes
    # Will be eliminated from the final code


    popinit(new_or_cont)
    # Initiate and store the Population Dictionary
    

    Gn=0
    # Generation Number
    # Counts down the generations


    Best=[fittest()]
    Gen=[Gn]
    Inp1=[[i, Gen, Pop, Fitness, Best] for i in range(0, Ps)]
    # Input 1
    # This input is repeatedly used to send to the parallely processed function
    # The inner components will be explained further in the function itself


    # Inpx=[[Pop[i], i, Gen] for i in range(Ps)]


    while Gn<Gs:
        Gn=Gn+1

    # Run the while loop Gs times
    # This imitates Gs Generations of Evolution

        # Fr[0]=Fr[0]/1.01
        # The above line of code can be used to change the value of F progressively

        Gen[0]= Gn


        with Pool(processes= Pc) as pool:
        # Create an instance of the pool process and call it "pool"

            result = pool.map_async(poprun, Inp1, chunksize= Ch)
            # For every population member, initiate poprun with parallel processing
            # The outputs will be stored in the generator "result"

            Temp=0
            # Temporary variable used for iterations as the variable i is occupied
            for Out in result.get():
            # ".get()" is used to extract outputs from a generator

                if Out!=0:
                # Function returns 0 if fitness of parent is greater

                    Pop[Temp]=Out[0]
                    # Change the population member to it's child

                    Fitness[Temp]=Out[1]
                    # Change the corresponding fitness to the child's fitness
                    
                Temp=Temp+1

            # If the Trial Vector is fitter than the population member, replace
                # the population member with the trial vector for the next generation,  
                # else do nothing

            # result = pool.map_async(fitnessFunction, Inpx, chunksize= Ch)

            # Temp=0
            # for Out in result.get():
                # if len(FitnessSOM)<Ps:
                    # FitnessSOM.append(Out)

                # if Out<FitnessSOM[Temp]:
                    # FitnessSOM[Temp]=Out
                # Temp=Temp+1
    

        Best[0]=fittest()
        print("Best= ", Best[0])

        Afit.append(sum(Fitness)/float(Ps))
        Bfit.append(Best[0][1])

        Temp=[]
        for i in range(Ps):
            Temp.append(Q(Pop[i]))

        Qval.append(Temp[Best[0][0]])
        Qavg.append(sum(Temp)/Ps)

        # AfitSOM.append(sum(FitnessSOM)/float(Ps))
        # BfitSOM.append(min(FitnessSOM))
        # QvalSOM.append(Q(Pop[FitnessSOM.index(BfitSOM[-1])]))


        if(Gn%5==0):
            plt.plot(range(0,len(Afit)), Afit, label="Avg Fitness")
            plt.plot(range(0,len(Bfit)), Bfit, label="Best Fitness")
            plt.plot(range(0,len(Qval)), Qval, label="Q value")
            plt.xlabel("Number of Generations")
            plt.ylabel("Fitness")
            plt.legend(loc="upper right")
            plt.savefig(f"graphs/Graph-Ps={Ps}-Gs={Gs}-DE.png")
            plt.close()
        

            with open("Values_DE.txt","w") as f:

                for i in range(len(Qval)):
                    print(Afit[i], file=f, end=",")
                    print(Bfit[i], file=f, end=",")
                    print(Qval[i], file=f, end=",")
                    print(Qavg[i], file=f, end="\n")
                    # print(AfitSOM[i], file=f, end=",")
                    # print(BfitSOM[i], file=f, end=",")
                    # print(QvalSOM[i], file=f, end="\n")



            if (Gn%50==0):
                for i in range(Ps):
                    Array= np.array(Pop[i], dtype="float32")
                    np.savez_compressed(f"./final_chromosomes/L{i}.npz", Array)

        if Gn!=Gs: 
            for i in range(Ps):

                if Gn%50==0 and i==Best[0][0]:
                    continue

                os.remove(f'./audio_output/gen{Gn}-{i}.wav')
                os.remove(f'./jAudio/gen{Gn}-{i}FV.xml')
                os.remove(f'./jAudio/gen{Gn}-{i}FK.xml')
                os.remove(f'./features_output/gen{Gn}-{i}_all_features.json')


    End= time.time()

    print(Best, End-Start)


if __name__ == '__main__':

    import os
    # This library is used to create/check folders/directories
    # creating output directories if they do not exist


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


    main()
