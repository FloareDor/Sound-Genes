import json
from audio_features import AudioFeatures
import sys
import math
import os
import pandas as pd
from decimal import Decimal, localcontext
import time
from Feature_Extractor import extract_features
from som.som_class import SOM



# normal distribution function
def g(x, mean, variance):
    return ((x - mean)**2) / (2 * variance)

# to add weights to the fitness values
def addWeights(fitnessValues, weights):
    weightesFitnessvalues = []
    for feature, fitnessValue in fitnessValues.items():
        weightesFitnessvalues.append(Decimal(weights[feature]) * Decimal(fitnessValue))
    
    # adding up the weighted fitness values
    fitnessValues["weightedSum"] = sum(weightesFitnessvalues)

    return fitnessValues

def computeFitnessValues(generation, populationNumber, audioFile="aaramb.wav", rasaNumber=1):

    output_folder = "features_output"

    startTime = time.time()
    directory = ""

    # Create an instance of the AudioFeatures class
    audio_features = AudioFeatures(directory = directory, filename = audioFile)

    # final data needed for the gaussian function
    gaussianData = pd.read_csv("updatedGaussianData.csv") 

    rasas = ['Karuna', 'Shanta', 'Shringar', 'Veera']

    weights = {
                "A1": 0.1,
                "A2": 0.1,
                "A3": 0.1,
                "A4": 0.1,
                "A5": 0.1,
                "B1": 0.1,
                "B2": 0.1,
                "B3": 0.1,
                "B4": 0.1,
                "B5": 0.1
            }
    rasa = rasas[0]
    # Call all the functions and store the results
    results = {
        "featureValues":{
            "A1": Decimal(audio_features.A1()),
            "A2": Decimal(audio_features.A2()),
            "A3": Decimal(audio_features.A3()),
            "A4": Decimal(audio_features.A4()),
            "A5": Decimal(audio_features.A5()),
            "B1": Decimal(audio_features.B1()),
            "B2": Decimal(audio_features.B2()),
            "B3": Decimal(audio_features.B3()),
            "B4": Decimal(audio_features.B4()),
            "B5": Decimal(audio_features.B5())
        },
    }

    fitnessValues = {}

    # to calculate fitness values from the feature values according to each rasa
    for i in range(len(rasas)):
        if rasaNumber == i+1 or rasaNumber==5:
            rasaValues = {}
            # print(results["featureValues"])
            index = 1
            for feature, value in results["featureValues"].items():
                mean = Decimal(gaussianData.iloc[index][f"{rasas[i]} Peak Point"])
                variance = Decimal(gaussianData.iloc[index]["5*SigmaSq obtained in Col G"])
                # calling the fitness function
                rasaValues[feature] = g(value, mean, variance)

                # print(feature, rasas[i], mean)
                index+=1

            rasaValues = addWeights(rasaValues, weights)
            fitnessValues[rasas[i]] = rasaValues

    results["fitnessValues"] = fitnessValues
    # print("The Fitness Function Execution Time is :", (time.time()-startTime), "s")

    # Save the results to a JSON file
    output_filename = f"{output_folder}/{audioFile.replace('.wav', '')}_10_features.json"
    with open(output_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4, default=str)

    try:
        
        os.remove(f'./jAudio/gen{generation}-{populationNumber}FV.xml')
        os.remove(f'./jAudio/gen{generation}-{populationNumber}FK.xml')

        if populationNumber % 25 != 0:
            os.remove(f'./audio_output/{audioFile}')
            os.remove(output_filename)
            os.remove(output_filename.replace("_10_", '_all_'))

        # print(f'Files successfully deleted.')
    except OSError as e:
        print(
           f'''
            Failed to delete the xml files: jAudio/gen{generation}-{populationNumber}FV.xml and jAudio/gen{generation}-{populationNumber}FK.xml
            error: {e}
            current path: {os.getcwd()}
            '''
            )
    
    # print(f"Results saved to {output_filename}")
    
    return results

def compute_SOM_DOB(SOM, generation, populationNumber, audioFile="Karuna.wav"):
    X=extract_features(filename=audioFile)

    print("number of extracted features:", len(X))

    prediction = SOM.predict(X)

    print("prediction: ", prediction)

    return prediction


if __name__ == "__main__":

    num_arguments = len(sys.argv)
    audioFile = "aaramb.wav"
    if num_arguments > 1:
        audioFile = sys.argv[1]
    if audioFile[-4:] != ".wav":
        audioFile+=".wav"

    results = computeFitnessValues(rasaNumber=5, audioFile=audioFile, generation=0, populationNumber=0)