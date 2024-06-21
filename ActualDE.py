from fourierSynthesis import *
import numpy as np
import os
import random
import copy
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# SOM FITNESS
def SOMfitness(Inp):
	# Inp = [Chromosome,generation,index,targetrasa]
	ifft_on_contracted_chromosome(Inp[0],32000,waves_per_bin=4,gen=Inp[1],candidateIndex=Inp[2])
	from som.som_class import SOM
	som = SOM()
	from driver import computeFitnessValues, compute_SOM_DOB
	generation=Inp[1]
	index=Inp[2]
	rasas = ['Karuna', 'Shanta', 'Shringar', 'Veera']
	targetRasa=rasas.index(Inp[3])
	rasaDob = compute_SOM_DOB(SOM=som,audioFile=f"gen{generation}-candidate{index}.wav", generation=generation, populationNumber=index)
	fitnessValue = rasaDob[targetRasa+1]
	return fitnessValue

def candidateGenerationFromExistingMusic(numberOfCandidates = 1,targetRasa = "Karuna",waves_per_bin = 4):
	BASE_FOLDER = 'C:\\Users\\shivq\\OneDrive\\Desktop\\DE Music\\Dataset'
	segment_duration = 0.5  # Duration of each Short Term Time Frame (STTF) in seconds
	waves_per_bin = 4 # Number of Waves in a Bin
	phase_type = 'first_STTF' # 
	candidates=[]
	for candidateIndex in range(0,numberOfCandidates):
		print(f"Generating Candidate {candidateIndex+1}")
		phaseFrames = []
		amplitudeFrames = []
		rasas = os.listdir(BASE_FOLDER)
		# GENERATING RANDOM WWEIGHTS WHICH SUM TO 1
		initializingWeights = np.random.dirichlet(np.ones(4),size=1)[0]
		# GENERATING RANDOM MUSIC FILE INDEX
		randomMusicIndex = np.random.randint(10, size=4)
		# GENERATING CHROMOSOME FROM ABOVE WEIGHTS AND MUSIC
		for rasa,rmi,iw in zip(rasas,randomMusicIndex,initializingWeights):
			allMusic = os.listdir(BASE_FOLDER+"\\"+rasa)
			musicFile = rasa+"\\"+allMusic[rmi]
			chromosome, song, sr = fft_on_wav(BASE_FOLDER, sample_song=musicFile, segment_duration=segment_duration)
			contracted_chromosome = contract_chromosome(chromosome, sr, song=None, waves_per_bin=waves_per_bin, phase_type=phase_type)
			# [[phases],[[frame1][frame2].......[frame60]]
            # MUTLTIPLYING A RANDOM WEIGHT TO CHROMOSOME
			for fid,frame in enumerate(contracted_chromosome[1]):
				contracted_chromosome[1][fid] = list(np.multiply(frame,iw))
			contracted_chromosome[0] = list(np.multiply(contracted_chromosome[0],iw))
            # CREATING LIST OF PHASE FRAMES FROM ALL 4 SELECTED COMPOSITIONS
			phaseFrames.append(contracted_chromosome[0])
			# CREATING LIST OF AMPLITUDES FRAMES FROM ALL 4 SELECTED COMPOSITIONS
			amplitudeFrames.append(contracted_chromosome[1])
		# CREATING A CANDIDATE FROM LIST OF PHASE FRAMES AND AMPLITUDE FRAMES
		ch = [np.sum(phaseFrames,axis=0),np.sum(amplitudeFrames,axis=0)]
		candidate = [ch,[SOMfitness([ch,0,candidateIndex,targetRasa])]]
		# ifft_on_contracted_chromosome(candidate,32000,waves_per_bin=4,gen=0,candidateIndex=0)
		candidates.append(candidate)
	
	return candidates

def candidateGenerationRandom(numberOfCandidates,targetRasa,numberOfPhases,numberOfFrames,numberOfBins,amplitudeRange = [0,600],phaseRange = [-180,180]):
	candidates=[]
	for candidateIndex in range(0,numberOfCandidates):
		print(f"Generating Random Candidate {candidateIndex+1}")
		candidate = []
		amplitudeFrames = [list(np.random.uniform(low=amplitudeRange[0], high=amplitudeRange[1], size=numberOfBins)) for _ in range(numberOfFrames)]
		phaseFrame = list(np.random.uniform(low=phaseRange[0], high=phaseRange[1], size=numberOfPhases))
		# Inp = [Chromosome,generation,index,targetrasa]
		candidate = [[phaseFrame,amplitudeFrames],[SOMfitness([[phaseFrame,amplitudeFrames],0,candidateIndex,targetRasa])]]
		print(candidate[1])
		candidates.append(candidate)    
	return candidates

def ComputeAmplitudeComponent(V1,V2,C):
	Vector = np.subtract(V1,V2)
	for fid,frame in enumerate(Vector):
		Vector[fid] = np.multiply(frame,C)
	return Vector

def ComputePhaseComponent(V1,V2,C):
	Vector = np.subtract(V1,V2)
	Vector = np.multiply(Vector,C)
	return Vector

def ChromosomeMutation(Chromosome,R1,R2,R3,K,F):
	# AMPLITUDE MUTATION
	component1 = ComputeAmplitudeComponent(R1[1],Chromosome[1],K)
	component2 = ComputeAmplitudeComponent(R2[1],R3[1],F)
	AmplitudeMutant = np.sum([Chromosome[1],component1,component2],axis=0)
	# PHASE MUTATION
	component1 = ComputePhaseComponent(R1[0],Chromosome[0],F)
	component2 = ComputePhaseComponent(R2[0],R3[0],F)
	PhaseMutant = np.sum([Chromosome[0],component1,component2],axis=0)
	# CLIPPING THE GENES IF REQUIRED
	# for frame in AmplitudeMutant:
	# 	Mutant.append(frame)
	# Mutant.append(list(PhaseMutant))
	# for fid,frame in enumerate(mutant):
	# 	for wid,wave in enumerate(frame):
	# 		if -5 <= wave[0] <= 5:
	# 			continue
	# 	else:
	# 		# wave[0] = np.clip(wave[0], -10, 10)
	# 		wave = Chromosome[fid][wid]
	return [PhaseMutant,AmplitudeMutant]

def ChromosomeCrossover(mutant,parent,crp):
	trialVector = copy.deepcopy(mutant)
	# AMPLITUDE CROSSOVER
	for fid,frame in enumerate(trialVector[1]):
		for bid, bin in enumerate(frame):
			if random.uniform(0,1) < crp:
				# trialVector[1][fid][bid] = trialVector[fid][bid]
				continue
			else:
				trialVector[1][fid][bid] = parent[1][fid][bid]
    # PHASE CROSSOVER
	for pid,phase in enumerate(trialVector[0]):
		if random.uniform(0,1) < crp:
			# trialVector[0][pid] = trialVector[0][pid]
			continue
		else:
			trialVector[0][pid] = parent[0][pid]
	return trialVector


Population = None
def GlobalDE(Inp):
		# [ChromosomeIndex,Chromosome[0],Chromosome[1],Population,CurrentGeneration]
		print(f"Generating Trial Chromosome: {Inp[0]}")
		K = 0.5 # K USED FOR DE
		CrossoverProbability = 0.9
		# FOR HALF OF THE POPULATION R1 IS BEST CHROMOSOME OF PREVIOUS GENERATION
		# R2 AND R3 ARE ALWAYS RANDOM
		if Inp[0] % 2 == 0:
			R1 = Inp[3][0]
			R2,R3 = random.sample(Inp[3][1:], 2)
			R1 = R1[0]
			R2 = R2[0]
			R3 = R3[0]
		# FOR HALF OF THE POPULATION R1 IS RANDOM
		# R2 AND R3 ARE ALWAYS RANDOM
		else:
			R1,R2,R3 = random.sample(Inp[3], 3)		
			R1 = R1[0]
			R2 = R2[0]
			R3 = R3[0]
		F = random.uniform(-2, 2)
		MutantChromsome = ChromosomeMutation(Inp[1],R1,R2,R3,K,F)
		TrialChromosome = ChromosomeCrossover(MutantChromsome,Inp[1],CrossoverProbability)
		ifft_on_contracted_chromosome(TrialChromosome,32000,waves_per_bin=4,gen=Inp[4],candidateIndex=Inp[0])
		ParentFitness = Inp[2]
		TrialFitness = SOMfitness([TrialChromosome,Inp[4],Inp[0],Inp[5]])# Inp = [Chromosome,generation,index,targetrasa]
		if TrialFitness < ParentFitness:
			return [TrialChromosome,[TrialFitness]]
		else:
			return [Inp[1],[ParentFitness]]

if __name__ == "__main__":
	waves_per_bin = 4
	segment_duration = 0.5
	music_duration = 30
	TargetRasa = "Karuna"
	# GENERATING POPULATION FROM EXISTING MUSIC 
	MusicPopulation = candidateGenerationFromExistingMusic(numberOfCandidates = 16,targetRasa = TargetRasa, waves_per_bin =4)
	shutil.rmtree("audio_output")
	if not os.path.exists("audio_output"):
		os.makedirs("audio_output")
	# GENERATING RANDOM POPULATION
	numberOfPhases = len(MusicPopulation[0][0][0])
	numberOfFrames = len(MusicPopulation[0][0][1])
	numberOfBins = len(MusicPopulation[0][0][1][0])
	RandomPopulation1 = candidateGenerationRandom(8,TargetRasa,numberOfPhases,numberOfFrames,numberOfBins,amplitudeRange = [-2,2],phaseRange = [-2,2])
	RandomPopulation2 = candidateGenerationRandom(4,TargetRasa,numberOfPhases,numberOfFrames,numberOfBins,amplitudeRange = [0,2],phaseRange = [0,2])
	RandomPopulation3 = candidateGenerationRandom(4,TargetRasa,numberOfPhases,numberOfFrames,numberOfBins,amplitudeRange = [-3,-1],phaseRange = [-3,1])
	# POPULATION COMBINATION OF RANDOM AND LINEAR SUM OF RANDOM EXISTING MUSIC CHROMOSOME
	Population = MusicPopulation + RandomPopulation1 + RandomPopulation2 + RandomPopulation3
	# Population = RandomPopulation1 + RandomPopulation2 + RandomPopulation3

    # DE PARAMETERS AND REQUIRED LIST DEFINATIONS
	CrossoverProbability = 0.9
	K = 0.5
	Generations = 50
	AverageFitnessAcrossGenerations = []
	GlobalBestFitnessAcrossGenerations = []
	CurrentGeneration = 0
	StopCount = 0	
	T1 = time.time()
	while CurrentGeneration < Generations:
		CurrentGeneration += 1
		print(f"Generation Cycle {CurrentGeneration}")
		# PAKAGING INPUTS FOR DE
		Inp = [[ChromosomeIndex,Chromosome[0],Chromosome[1][0],Population,CurrentGeneration,TargetRasa] for ChromosomeIndex, Chromosome in enumerate(Population)]
		# MULTIPROCESSING CODE 
		NextGeneration = []
		Population = []
		AllFitness = []
		with ThreadPoolExecutor(max_workers = 8) as executor:
			NextGeneration = executor.map(GlobalDE,Inp,chunksize=4)
		for _ in NextGeneration:
			Population.append(_)
			AllFitness.append(_[1][0])

		#SORTING POPULATION AS PER FITNESS
		AllFitness = np.array(AllFitness)
		sortedArgs = list(AllFitness.argsort())
		SortedPopulation = [Population[rank] for rank in sortedArgs]
		Population = copy.copy(SortedPopulation)
		bestFitness = AllFitness[0]
		# SAVING BEST POPULATION
		for _ in Population[0:1]:
			bestIFFT_on_contracted_chromosome(_[0],32000,waves_per_bin=4,gen=CurrentGeneration,candidateIndex=_[1][0])
		GlobalBestFitnessAcrossGenerations.append(bestFitness)
		AverageFitnessAcrossGenerations.append(np.average(AllFitness[:int(len(AllFitness)/2)]))
		print(f'Best Fitness Generation {CurrentGeneration} = {bestFitness}')
		print(f'Average Fitness Generation {CurrentGeneration} = {np.average(AllFitness[:int(len(AllFitness)/2)])}')
		shutil.rmtree("audio_output")
		if not os.path.exists("audio_output"):
			os.makedirs("audio_output")
		shutil.rmtree("features_output")
		if not os.path.exists("features_output"):
			os.makedirs("features_output")
		# HERE WE ARE DISCARDING THE BOTTOM HALF CHROMOSOMES AND RE GENERATING NEW RANDOM CHROMOSOMES 
		if CurrentGeneration%10 == 0:
			Population = Population[0:-16]
			NewPopulation1 = candidateGenerationRandom(8,TargetRasa,numberOfPhases,numberOfFrames,numberOfBins,amplitudeRange = [-2,2],phaseRange = [-2,2])
			NewPopulation2 = candidateGenerationRandom(4,TargetRasa,numberOfPhases,numberOfFrames,numberOfBins,amplitudeRange = [0,2],phaseRange = [0,2])
			NewPopulation3 = candidateGenerationRandom(4,TargetRasa,numberOfPhases,numberOfFrames,numberOfBins,amplitudeRange = [-3,-1],phaseRange = [-3,-1])
			Population = Population + NewPopulation1 + NewPopulation2 + NewPopulation3

	T2 = time.time()
	print(f"DE DURATION {T2-T1}")
	# SIMPLE GRAPH PLOT FITNESS (AVERAGE & BEST) VS GENERATIONS 
	import matplotlib.pyplot as plt
	plt.rcParams.update({'font.size': 18})
	fig = plt.figure(figsize=(12,6))
	plt.plot(GlobalBestFitnessAcrossGenerations, color='blue', label='Best Fitness')
	plt.plot(AverageFitnessAcrossGenerations, color='Red', label='Average Fitness')
	plt.xlabel('Generations',fontsize=18) 
	plt.ylabel('Fitness',fontsize=18) 
	plt.title(f'DE | Values = {numberOfPhases+numberOfFrames*numberOfBins} | Time = {T2-T1} | Population = {len(Population)}', fontsize=22) 
	plt.grid(True) 
	plt.legend(loc="upper left",fontsize=18)
	plt.show()

					
				




          
          




