import random as rd
# This is used to for random number generation wherever neccessary

from copy import deepcopy
# This is a crucial library used to create perfect copies of chromosomes

from matplotlib import pyplot as plt

from chromosome_processor import decode

from driver import computeFitnessValues

A=10
# Amplitude
# This is the maximum amplitude or loudness

Cp=0.8
# Crossover Probability
# This is the amount of crossover between the original vector and the mutant vector 
    # and is used to create the trial vector

K=0.8
# Coefficient used to generate the mutant vector

Fr=[2]
# Coefficient used to generate the mutant vector

# Comment by Arya added 07092023. It may be advisable to reduce K to 0.5 and range of Fr to 1 instead of 2. 

Ps= 5
# Population Size

Gs= 5
# Generation Size

Cl= 100
# Chromosome Length
# This is the number of Time Frames in a single song

Gl= 1
# Gene Length
# This is the number of frequency Bins in a single Time Frame

Pc= 2
# Number of processes
# This is the number of cores this code should parallely run on

Ch= 100//Pc+1
# Chunk size
# This is the number of elements each parallel run should process before returning

def ff(L):
# Fitness Function
# This must be minimised in the given situation

    for i in range(0, 1):
    # The fitness function is being revaluated here
                
        E=0
        # Error
        # This will be used to evaluate the fitness of a chromosome

        for j in range(Cl):
            for k in range(Gl):
                for l in range(2):
                    E= E + abs(L[j][k][l]-Test[j][k][l])

    # The error is the manhattan distance of a chromosome from chromosome Test
    # Hence the algorithm must minimise this error

    return E

# The original fitness function (integrated)
def fitnessFunction(chromosome, index, rasaNumber=1):

    rasas = ['Karuna', 'Shanta', 'Shringar', 'Veera']
    chromosome_copy = deepcopy(chromosome)

    decode(chromosome_copy, f"test_data/{index}.wav")

    values = dict(computeFitnessValues(rasaNumber=rasaNumber, audioFile=f"{index}.wav"))
    fitnessValue = float(values["fitnessValues"][rasas[rasaNumber-1]]["weightedSum"])

    return fitnessValue

def chrm():
# Random Chromosome Generator
# Creates a random chromosome with the permissible boundary values

    L=[]

    for i in range(Cl):
        L.append([])
        
        for j in range(Gl):
            L[i].append([rd.uniform(0,A), rd.uniform(20,2000)])
    
    return L


def popinit():
# Population Initiator
# Creates an initial population pool

    for i in range(Ps):
        Pop[i]=chrm()
        Fitness.append(fitnessFunction(Pop[i], i))
        
    # Every element in Pop is a Chromosome indexed from an integer from 0 to Ps

    # print(f"Pop: {Pop}")
    return Pop


def fittest():
# Fittest Population Member Evaluator
# Finds the Population Member with the Greatest Fitness and that Fitness

    Bf=10000000
    # Best Fitness
    # This the Best Fitness found
    # Initially it is set to an arbitrarily high number for minimization

    Fiti=0
    # Fittest Member Index
    # This is the chromosome index with the highest fitness
    # Initially it is the first chromosome

    for i in range(Ps):
        if Fitness[i]<Bf:
            Bf= Fitness[i]
            Fiti=i
        
        # Simply keep track of the lowest error among the population members
            
    # Bf is now the best fitness and Fiti is the corresponding population member

    return Fiti, Bf


def poprun(Inp):



    i=Inp[0]
    # 1) This is the index of the chromosome this call must work on
    Pop= Inp[1]
    # 2) This is the population dictionary that contains all chromosomes
    global Test
    Test=Inp[2]
    # 3) This is the Test chromosome
       # It has been sent here so that the fitness function can be called directly 
    X=Inp[3][0][0]
    # 4) This is the fittest chromosome from the previous generation
       # It is used for Global search (Minimal additional cost)
    Fiti=Inp[4][i]
    # 5) This is the fitness of the current chromosome
       # It is not evaluated natively to reduce number of fitness function evaluations


    for z in range(0,10):
        # print(f"z: {z}")
    # Maximum number of times the chromosome should be revaluated if it is out of bounds

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
                for l in range(2):
                    Mut[j][k][l]= Mut[j][k][l]+K*(Pop[X][j][k][l]-Pop[i][j][k][l])+F*(Pop[y][j][k][l]-Pop[z][j][k][l])

                    # The above is a direct formula used to create mutant vectors in DE

        # Mut is now the Mutant Vector of population member L

        Tri=deepcopy(Mut)
        # Trial Vector
        # Tri is the Trial Vector of the population member L
        # Initiate it to be the Mutant Vector

        for j in range(Cl):
            for k in range(Gl):
                for l in range(2):
                    if (rd.uniform(0,1)> Cp):
                        Tri[j][k][l]= Pop[i][j][k][l]
                    
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
                    
                if (Tri[j][k][1]>16000 or Tri[j][k][1]<0):
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
                    
                if (Tri[j][k][1]>16000 or Tri[j][k][1]<0):

                    Tri[j][k][1]=Pop[i][j][k][1]

    # If a component of the Trial Vector is Violating a constraint, replace that 
        # component with that of the population member

    Temp=fitnessFunction(Tri, "trial")
    # Temp is used here to reduce the number of fitness function calls

    if(Temp<Fiti):
        return [Tri, Temp]
        # If successful, return the child chromosome as well as it's fitness
        
    else:
        return 0
        # If failure, return 0
    

def main():

    from multiprocessing import Pool
    # This is a crucial library used to implement parallel processing in the algorithm

    import time
    # This is simply used to calculate the total execution time of the code

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

    global Test
    Test=[]
    # Test Chromosome
    # Let us assume that this chromosome is the perfect song
    # The goal of our algorithm then is to achieve the closest chromosome that
        # replicates this


    for i in range(Cl):
        Test.append([])
        
        for j in range(Gl):
            Test[i].append([rd.uniform(0,A), rd.uniform(0,360)])

    # Initiation of Test Chromosome
    popinit()
    # Initiate and store the Population Dictionary

    Gc=0
    # Generation Counter
    # Counts down the generations

    Gc=Gs

    Best=[fittest()]
    # Used in the next line of code
    Inp1=[[i, Pop, Test, Best, Fitness] for i in range(0, Ps)]
    # print(f"Pop: {Pop}")
    # print(f"Test: {Test}")
    # print(f"Best: {Best}")
    # print(f"Fitness: {Fitness}")
    # Input 1
    # This input is repeatedly used to send to the parallely processed function
    # The inner components will be explained further in the function itself

    while Gc>0:
        print(f"Gc: {Gc}/{Gs}")
        Gc=Gc-1
    # Run the while loop Gs times
    # This imitates Gs Generations of Evolution

        # Fr[0]=Fr[0]/1.01x``
        # The above line of code can be used to change the value of F progressively

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
    
        Best[0]=fittest()
        Bfit.append(Best[0][1])
        # Append the highest fitness population member of each generation for
            # plotting purposes
        # Will be eliminated from the final code

    End= time.time()

    print(Bfit[-1], End-Start)
    # Print the error between the Test chromosome and the Final chromosome

    

    plt.plot(range(0,len(Bfit)), Bfit)
    plt.xlabel("Number of Generations")
    plt.ylabel("Fitness")
    plt.show()

if __name__ == '__main__':

    main()