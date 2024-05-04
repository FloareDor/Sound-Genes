import numpy as np

import random as rd
# This is used to for random number generation wherever neccessary

from copy import deepcopy
# This is a crucial library used to create perfect copies of chromosomes

import os   
# This library is used to create/check folders/directories
# creating output directories if they do not exist

### HYPERPARAMETERS

A= 10
# Amplitude
# This is the maximum amplitude or loudness

Cp= 0.9
# Crossover Probability
# This is the amount of crossover between the original vector and the mutant
    # vector used to create the trial vector

K=0.7
# Coefficient used to generate the mutant vector

Fr=[3]
# Coefficient used to generate the mutant vector

Gc= 50
# Hyperparameter used for cyclical variation in local mutation
# Must not be 0


### EVOLUTION PARAMETERS

Ps= 5
# Population Size

Gs= 11
# Generation Size

new_or_cont= 0

Lt= 0
# Local Search Threshold
# Above this value, the search will incorporate local search as well

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
# This is the number of elements each parallel run should process before returning


### MUSIC PARAMETERS

Cp= 10
# Compression Parameter
# This is the number of bins to be compressed into one bin in after compression.

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

    chromosome=Inp[0]
    index=Inp[1]
    generation=Inp[2]
    rasaNumber=1

    rasas = ['Karuna', 'Shanta', 'Shringar', 'Veera']
    chromosome_copy = deepcopy(chromosome).tolist()

    for j in range(1,Cl):
        for k in range(Gl):
            chromosome_copy[j][k].append(chromosome_copy[0][k][1])

    decode(chromosome_copy, index=index, generation=generation, Minfrq=Minfrq, Maxfrq=Maxfrq, Cl=Cl, Gl=Gl, Wpb=Wpb, TS=Srate*60, Srate=Srate)

    return


def Q(L):
    
    sum= 0

    for j in range(Cl):
        for k in range(L.shape[1]-1):

            sum+= abs(L[j][k][0]-L[j][k+1][0])

        if j!=Cl-1:
            sum+=abs(L[j][0][0]-L[j+1][k-1][0])

    return sum/2000


def ff(Inp):

    L=Inp[0]
    Gn=Inp[2]

    if Gn< St:
        Test= np.load(f'./initial_chromosomes/L1000cc.npz')
        Test=Test['arr_0']
    else:
        Test= np.load(f'./initial_chromosomes/L1000c.npz')
        Test=Test['arr_0']

    if Gn>=Gs:
        Test= np.load(f'./initial_chromosomes/L1000.npz')
        Test=Test['arr_0']

    return [np.mean(np.square(Test-L)), 0, Q(L)]


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


def encoder(L, compression):
    '''
    Takes a chromosome L and returns a compressed version where every k bins are averaged
    to one bin. k= Compression in input.
    '''

    T=[]
    
    for i in range(Cl):
        T.append([])

        for j in range(0, L.shape[1], compression):
            T[i].append(np.average(L[0][j:j+10], axis=0))

    T=np.array(T)

    return T

def distribution(Initial, compression, Final):
    '''
    Takes a chromosome L and returns the distribution of the chromosome.
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
        Test= np.load(f'./initial_chromosomes/L1000.npz')
        Test=Test['arr_0']
    else:
        Test= np.load(f'./initial_chromosomes/L1000c.npz')
        Test=Test['arr_0']

    Testcompressed= encoder(Test, compression=Cp)

    if callno==1:
        np.savez_compressed(f"./initial_chromosomes/L1000c.npz", Testcompressed)
    else:
        np.savez_compressed(f"./initial_chromosomes/L1000cc.npz", Testcompressed)

    Distribution= distribution(Test, Cp, Testcompressed)


    for i in range(Ps):
        Pop[i]= encoder(Pop[i], compression= Cp)

    return Distribution

def decodeRoutine(Distribution):

    for i in range(Ps):
        Pop[i]= decoder(Pop[i], compression= Cp, Distribution= Distribution)

    return


def popinit(new_or_cont):
# Population Initiator
# Creates an initial population pool

    from multiprocessing import Pool

    for i in range(Ps):
        Pop[i]=chrm(new_or_cont, i)

    Distributionc= encodeRoutine(1)
    Distributioncc= encodeRoutine(2)
        
    Inp=[[Pop[i], i, 0] for i in range(Ps)]

    with Pool(processes= Pc) as pool:

        result = pool.map_async(ff, Inp, chunksize= Ch)

        for Out in result.get():
            Fitness.append(Out[0])
            SFitness.append(Out[1])
            QFitness.append(Out[2])

    # Every element in Pop is a Chromosome indexed from an integer from 0 to Ps

    return [Distributionc, Distributioncc]


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

        F=rd.uniform(-Fr[0]/f, Fr[0]/f)
        # Coefficient used to generate the mutant vector

        while(1):
            
            y,z= rd.sample(range(Ps),2)

            if(y!=i and z!=i and y!=X and z!=X):
                break
            
        # The above are used to choose population vectors to mutate the original 
            # vector with
        
        Mut+= K*(Pop[X]-Pop[i])+F*(Pop[y]-Pop[z])
        # There's a reason it is done in the above way!!

        # Mut is now the Mutant Vector of population member L

        Temp= 0

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


    i=Inp[0]

    if i==0:
        return 0

    Gn=Inp[1][0]
    Pop=Inp[2]
    # Fiti requires the arguments from Inp[4]
    Ind1=Inp[4][i]
    Ind2=Inp[4][i-1]

    Fiti=Inp[3][Ind2]


    Cro=deepcopy(Pop[Ind1])

    R=rd.uniform(0,1)

    Cro+= R*(Pop[Ind2]-Cro)


    Temp=ff([Cro, i, Gn])

    if(Temp[0]<=Fiti):
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

    Inp1= Inp[0]
    Inp2= Inp[1]
    Inp3= Inp[2]
    Inp4= Inp[3]


    i=Inp1[0]
    Pop=Inp1[2]
    Fitness=Inp1[3]
    Gn=Inp1[1][0]
    Indsort=Inp2[4]
    SFitness=Inp4[0]
    QFitness=Inp4[1]


    if Gn>Lt:

        Out= loccro(Inp2)

        if Out!=0:

            Pop[Indsort[i]]=Out[0]

            Fitness[Indsort[i]]=Out[1][0]
            SFitness[Indsort[i]]=Out[1][1]
            QFitness[Indsort[i]]=Out[1][2]


        Out= locmut(Inp3)
        if Out!=0:

            Pop[i]=Out[0]

            Fitness[i]=Out[1][0]
            SFitness[i]=Out[1][1]
            QFitness[i]=Out[1][2]


    Out= poprun(Inp1)
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


    manager= Manager()

    global Pop
    Pop= manager.dict()
    # Population Dictionary
    # This is a dictionary mapping integers to a unique Chromosome
    # Initiate it as an empty dictionary

    global Fitness
    Fitness= manager.list()
    # Fitness List
    # This is a list that stores the fitness of every population member at any given time
    # Initiate it as an empty list

    global Bfit
    Bfit=[]
    # Best Fitness List
    # This is a list of the best fitness in every generation for plotting purposes

    Afit=[]
    # Average Fitness List
    # This is a list of the average fitness in every generation for plotting purposes

    global SFitness
    SFitness= manager.list()
    # SOM Fitness of all population members

    Sval=[]
    # SOM Fitness of fittest population member
    Savg=[]

    global QFitness
    QFitness= manager.list()
    # Q Fitness of all population members

    Qval=[]
    # Q Fitness of fittest population member
    Qavg=[]


    Out=popinit(new_or_cont)
    Distributionc= Out[0]
    Distributioncc= Out[1]
    # Initiate and store the Population Dictionary
    

    Gn=0
    # Generation Number
    # Counts down the generations


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


    while Gn<Gs:
        Gn=Gn+1
        print("Gen: ", Gn)
        Gen[0]= Gn
        Cyc[0]= np.exp(-2*(Gn%Gc)/Gc)* 1/(1+np.exp(-1*((Gc/2)-(Gn%Gc))))


        if Gn==St:
            decodeRoutine(Distributioncc)
        if Gn== Gs:
            decodeRoutine(Distributionc)


        with Pool(processes= Pc) as pool:
        # Create an instance of the pool process and call it "pool"

            if Gn>Lt:
            # If Local Search Threshold is surpassed, perform Local Search as well

                Junk,Temp=zip(*sorted(zip(Fitness, Ind)))
                # Sorted in descending order as lower fitness is better

                for i in range(0, Ps):
                    Indsort[i]=Temp[i]

                    Pm[Temp[i]]=i/Ps

                if Gn%10==1 or Gn==St or Gn==Gs-1:

                    Popcopy=deepcopy(Pop)

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
            # For every population member, initiate poprun with parallel processing
            # The outputs will be stored in the generator "result"

            for _ in result.get():
            # ".get()" is used to extract outputs from a generator
                
                pass


        Best[0]=fittest()

        Afit.append(sum(Fitness)/float(Ps))
        Bfit.append(Best[0][1])

        Sval.append(SFitness[Best[0][0]])
        Savg.append(sum(SFitness)/Ps)

        Qval.append(QFitness[Best[0][0]])
        Qavg.append(sum(QFitness)/Ps)

        print("Best= ", Best[0], Sval[-1], Qval[-1])


        if(Gn%5==0):
            plt.plot(range(0,len(Afit)), Afit, label="Avg Fitness")
            plt.plot(range(0,len(Bfit)), Bfit, label="Best Fitness")
            plt.xlabel("Number of Generations")
            plt.ylabel("Fitness")
            plt.legend(loc="upper right")
            plt.savefig(f"graphs/Graph.png")
            plt.close()
        

            with open("Values_LSDE.txt","w") as f:

                for i in range(len(Qval)):
                    print(Afit[i], file=f, end=",")
                    print(Bfit[i], file=f, end=",")
                    print(Sval[i], file=f, end=",")
                    print(Savg[i], file=f, end=",")
                    print(Qval[i], file=f, end=",")
                    print(Qavg[i], file=f, end="\n")


            if (Gn%10==0):
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
                


    for i in range(Ps):
        Array= np.array(Pop[i], dtype="float32")
        np.savez_compressed(f"./final_chromosomes/L{i}.npz", Array)

    End= time.time()

    print(Best, End-Start)


if __name__ == '__main__':

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