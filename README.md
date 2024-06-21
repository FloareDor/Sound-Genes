
# Project Title




## Documentation
Target Rasa

    TargetRasa = "Karuna"
        this can be any of these 4 = ['Karuna', 'Shanta', 'Shringar', 'Veera']

Generating chromosome from linear interpolation of exisiting music

        MusicPopulation = candidateGenerationFromExistingMusic(numberOfCandidates = 1,targetRasa = "Karuna",waves_per_bin = 4)

        Popuation = MusicPopulation 
        (take care that you comment the lines which are used to random population generation) 
        (if you only want population to be linear interpolation)
        (IMPORTANT : BASE_FOLDER = 'C:\\Users\\shivq\\OneDrive\\Desktop\\DE Music\\Dataset' CHANGE THIS AS PER THE SERVER)

Changing number of threads to be used
 
        with ThreadPoolExecutor(max_workers = 8) as executor:
			    NextGeneration = executor.map(GlobalDE,Inp,chunksize=4)

        ( change max workers to change number of threads and chunksize = Total Population / max_workers )
        ( keep in mind every core has 2 logical threads ) ( do not max out your thread count it may lead to unstability )

DE PARAMETERS
    
    CrossoverProbability = 0.9
	K = 0.5
	Generations = 50

Requirements

    java jdk22
    python 3.12.2
    numpy





