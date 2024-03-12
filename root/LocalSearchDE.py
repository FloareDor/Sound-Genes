import random as rd
# This is used to for random number generation wherever neccessary

from copy import deepcopy
# This is a crucial library used to create perfect copies of chromosomes


### HYPERPARAMETERS

A=10
# Amplitude
# This is the maximum amplitude or loudness

Cp=0.8
# Crossover Probability
# This is the amount of crossover between the original vector and the mutant
    # vector used to create the trial vector

K=0.5
# Coefficient used to generate the mutant vector

Fr=[2]
# Coefficient used to generate the mutant vector

Gc= 50
# Hyperparameter used for cyclical variation in local mutation
# Must not be 0


### EVOLUTION PARAMETERS

Ps= 5
# Population Size

Gs= 5
# Generation Size

Lt= 0
# Local Search Threshold
# Above this value, the search will incorporate local search as well


### MULTIPROCESSING ARGUMENTS

Pc= 5
# Number of processes
# This is the number of cores this code should parallely run on

Ch= 1
# Chunk size
# This is the number of elements each parallel run should process before returning


### MUSIC PARAMETERS

Cl= 239
# Chromosome Length
# Frames per audio sample
# This is the number of Time Frames in a single song

Gl= 329
# Gene Length
# Bins per frame
# This is the number of frequency Bins in a single Time Frame

Wpb= 24
# Waves per bin
# This is the number of waves in a single Bin


Frl=0.5
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
ff1i=34 # Average value of ff1
ff2i=75 # Average value of ff1
s1=1
s2=ff1i/ff2i
w1=0.95 # Weight of ff1
w2=0.05 # Weight of ff2


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

    from mel_chr_to_wav import decode
    # This is the function that converts chromosomes to wav files.

    from driver import computeFitnessValues, compute_SOM_DOB
    # This is the fitness function that utilizes 5 features from set A and another 5 from set B


    chromosome=Inp[0]
    index=Inp[1]
    generation=Inp[2]
    rasaNumber=1

    rasas = ['Karuna', 'Shanta', 'Shringar', 'Veera']
    chromosome_copy = deepcopy(chromosome)

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


def Q(L):
    
    sum=0

    for j in range(Cl):
        for k in range(Gl-1):

            sum+= abs(L[j][k][0]-L[j][k+1][0])

        if j!=Cl-1:
            sum+=abs(L[j][0][0]-L[j+1][k-1][0])

    return sum/2000


### ASSISTING FUNCTIONS

def chrm():
# Random Chromosome Generator
# Creates a random chromosome with the permissible boundary values

    L=[]

    for i in range(Cl):
        L.append([])
        
        for j in range(Gl):

            if i==0:
                L[i].append([rd.uniform(0,A), rd.uniform(0,6.282)])
            else:
                L[i].append([rd.uniform(0,A)])
    
    return L


def popinit():
# Population Initiator
# Creates an initial population pool

    from multiprocessing import Pool

    for i in range(Ps):
        Pop[i]=chrm()
        
    Inp=[[Pop[i], i, 0] for i in range(0, Ps)]

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


    i=Inp[0]
    # 1) This is the index of the chromosome this call must work on
    Gn = Inp[1][0]
    # 2) Generation Number
        # This is used in the fitness function
    Pop= Inp[2]
    # 3) This is the population dictionary that contains all chromosomes
    Fiti=Inp[3][i]
    # 4) This is the fitness of the current chromosome
       # It is not evaluated natively to reduce number of fitness function evaluations
    X=Inp[4][0][0]
    # 5) This is the fittest chromosome from the previous generation
       # It is used for Global search (Minimal additional cost)
   

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
        
        for j in range(Cl):
            for k in range(Gl):

                if j==0:
                    for l in range(2):
                        Mut[j][k][l]= Mut[j][k][l]+K*(Pop[X][j][k][l]-Pop[i][j][k][l])+F*(Pop[y][j][k][l]-Pop[z][j][k][l])
                else:
                    Mut[j][k][0]= Mut[j][k][0]+K*(Pop[X][j][k][0]-Pop[i][j][k][0])+F*(Pop[y][j][k][0]-Pop[z][j][k][0])

                    # The above is a direct formula used to create mutant vectors in DE

        # Mut is now the Mutant Vector of population member L

        Tri=deepcopy(Mut)
        # Trial Vector
        # Tri is the Trial Vector of the population member L
        # Initiate it to be the Mutant Vector

        for j in range(Cl):
            for k in range(Gl):

                if j==0:
                    for l in range(2):
                        if (rd.uniform(0,1)> Cp):
                            Tri[j][k][l]= Pop[i][j][k][l]

                else:
                    if (rd.uniform(0,1)> Cp):
                        Tri[j][k][0]= Pop[i][j][k][0]

                    # Here as the Tri is the same as Mut initially, elements of Tri
                        # are restored to the value of L with a probability of 1-Cp

        # Tri is now the Trial Vector of the population member L

        Temp= 0
        # Temporary Variable
        # Used for a variety of basic tasks

        for j in range(Cl):
            for k in range(Gl):

                if (Tri[j][k][0]> A or Tri[j][k][0]<0):
                    Temp= 1
                    
                if j==0:
                    if (Tri[j][k][1]>6.282 or Tri[j][k][1]<0):
                        Temp= 1

                if Temp==1:
                    break

            if Temp==1:
                break
        
        if Temp==0:
            break

        # If none of the constraints are violated (Temp==0) then break, else revaluate
            # from the mutation vector


    if Temp!=0:
        for j in range(Cl):
            for k in range(Gl):

                if (Tri[j][k][0]> A or Tri[j][k][0]<0):

                    Tri[j][k][0]=Pop[i][j][k][0]
                    
                if j==0:
                    if (Tri[j][k][1]>6.282 or Tri[j][k][1]<0):

                        Tri[j][k][1]=Pop[i][j][k][1]

    # If a component of the Trial Vector is Violating a constraint, replace
        # that component with that of the population member

    Temp=fitnessFunction([Tri, i, Gn])
    # Temp is used here to reduce the number of fitness function calls

    if(Temp<=Fiti):
        return [Tri, Temp]
        # If successful, return the child chromosome as well as it's fitness

    else:
        return 0
        # If failure, return 0
 

def loccro(Inp):


    i=Inp[0]
    Gn=Inp[1][0]
    Pop=Inp[2]
    # Fiti requires the arguments from Inp[4]
    Ind1=Inp[4][i]
    Ind2=Inp[4][i+1]

    Fiti=Inp[3][Ind2]


    Cro=deepcopy(Pop[Ind1])

    R=rd.uniform(0,1)

    for j in range(Cl):
        for k in range(Gl):
            
            if j==0:
                for l in range(2):
                    Cro[j][k][l]= Cro[j][k][l]+R*(Pop[Ind2][j][k][l]-Cro[j][k][l])
            
            else:
                Cro[j][k][0]= Cro[j][k][0]+R*(Pop[Ind2][j][k][0]-Cro[j][k][0])


    Temp=fitnessFunction([Cro, i, Gn])

    if(Temp<=Fiti):
        return [Cro, Temp]
        # If successful, return the child chromosome as well as it's fitness

    else:
        return 0
        # If failure, return 0


def locmut(Inp):


    i=Inp[0]
    Gn=Inp[1][0]
    Pop=Inp[2]
    Fiti=Inp[3][i]
    Min=Inp[4][0]
    Max=Inp[4][1]
    Pm=Inp[4][2][i] # This is ready for multiplication, not done n-i way.
    Cyc=Inp[4][3][0]


    Mut=deepcopy(Pop[i])


    for j in range(Cl):
        for k in range(Gl):
            
            R=rd.uniform(0,1)

            if j==0:
                for l in range(2):
                
                    x=rd.randint(0,1)

                    if x==0:
                        Mut[j][k][l]= Mut[j][k][l]+R*Cyc*Pm*(Max[j][k][l]-Mut[j][k][l])
                    else:
                        Mut[j][k][l]= Mut[j][k][l]+R*Cyc*Pm*(Min[j][k][l]-Mut[j][k][l])
            
            else:
                x=rd.randint(0,1)

                if x==0:
                    Mut[j][k][0]= Mut[j][k][0]+R*Cyc*Pm*(Max[j][k][0]-Mut[j][k][0])
                else:
                    Mut[j][k][0]= Mut[j][k][0]+R*Cyc*Pm*(Min[j][k][0]-Mut[j][k][0])


    Temp=fitnessFunction([Mut, i, Gn])

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
    # Q Values List
    # This is a list of the Q values in every generation for plotting purposes
    # Will be eliminated from the final code


    popinit()
    # Initiate and store the Population Dictionary
    

    Gn=0
    # Generation Number
    # Counts down the generations


    Best=[fittest()]
    Gen=[Gn,Gc]
    Inp1=[[i, Gen, Pop, Fitness, Best] for i in range(0, Ps)]

    Ind=[i for i in range(0, Ps)]
    Indsort=deepcopy(Ind)
    Inp2=[[i, Gen, Pop, Fitness, Indsort] for i in range(0, Ps-1)]

    Min=deepcopy(Pop[0])
    Max=deepcopy(Pop[0])
    Pm=[0]*Ps
    Cyc=[0]
    Inp3=[[i, Gen, Pop, Fitness, [Min, Max, Pm, Cyc]] for i in range(0, Ps)]


    while Gn<Gs:
        Gn=Gn+1

    # Run the while loop Gs times
    # This imitates Gs Generations of Evolution

        # Fr[0]=Fr[0]/1.01
        # The above line of code can be used to change the value of F progressively

        Gen[0]= Gn
        Cyc[0]= np.exp(-2*Gn/Gc)* 1/(1+np.exp(-1*((Gc/2)-Gn)))


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
                # the population member with the trial vector for the next 
                # generation, else do nothing

            if Gn>Lt:
            # If Local Search Threshold is surpassed, perform Local Search as well


                Junk,Temp=zip(*sorted(zip(Fitness, Ind)))
                # Sorted in descending order as lower fitness is better

                for i in range(0, Ps):
                    Indsort[i]=Temp[i]
                    # The above step is crucial so that Indsort in Inp2 does not get changed

                    Pm[Temp[i]]=i


                result = pool.map_async(loccro, Inp2, chunksize=Ch)

                Temp=1
                for Out in result.get():

                    if Out!=0:
                        Pop[Indsort[Temp]]=Out[0]
                        Fitness[Indsort[Temp]]=Out[1]
                    
                    Temp=Temp+1


                for i in range(Ps):
                    for j in range(Cl):
                        for k in range(Gl):

                            if j==0:
                                for l in range(2):
                                    if (Pop[i][j][k][l]<Min[j][k][l]):
                                        Min[j][k][l]= Pop[i][j][k][l]
                        
                                    elif(Pop[i][j][k][l]>Max[j][k][l]):
                                        Max[j][k][l]= Pop[i][j][k][l]

                            else:
                                if (Pop[i][j][k][0]<Min[j][k][0]):
                                    Min[j][k][0]= Pop[i][j][k][0]
                        
                                elif(Pop[i][j][k][0]>Max[j][k][0]):
                                    Max[j][k][0]= Pop[i][j][k][0]


                result = pool.map_async(locmut, Inp3, chunksize=Ch)

                Temp=0
                for Out in result.get():

                    if Out!=0:
                        Pop[Temp]=Out[0]
                        Fitness[Temp]=Out[1]
                    
                    Temp=Temp+1


        Best[0]=fittest()
        print("Best= ", Best[0])

        Afit.append(sum(Fitness)/float(Ps))
        Bfit.append(Best[0][1])
        Qval.append(Q(Pop[Best[0][0]]))


        if(Gn%5==0):
            plt.plot(range(0,len(Afit)), Afit, label="Avg Fitness")
            plt.plot(range(0,len(Bfit)), Bfit, label="Best Fitness")
            plt.plot(range(0,len(Qval)), Qval, label="Q value")
            plt.xlabel("Number of Generations")
            plt.ylabel("Fitness")
            plt.legend(loc="upper right")
            plt.savefig(f"graphs/Graph-Ps={Ps}-Gs={Gs}-LSDE.png")
            plt.close()
        

            with open("Values_LSDE.txt","w") as f:

                for i in range(len(Qval)):
                    print(Afit[i], file=f, end=",")
                    print(Bfit[i], file=f, end=",")
                    print(Qval[i], file=f, end="\n")

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

    main()
