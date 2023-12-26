import json
import xml.etree.ElementTree as ET
import subprocess
from decimal import Decimal
import os
import coreFeatures
import numpy as np

def extract_features(settings="final-104.xml", directory="", filename="aaramb.wav", output_filename=None):

    output_folder = "features_output"
    # If no output filename is provided, use the original filename without .wav extension
    if output_filename is None:
        output_filename = filename.replace(".wav", "")
    
    # Create the full audio file path
    audio_filepath = os.path.join(directory, filename)
    
    # Construct the Java command to run jAudio feature extraction
    java_command = [
        'java',
        '-Xmx1024M',
        '-jar',
        'jAudio.jar',  # jAudio Executable filename
        '-s',
        settings,  # jAudio settings file
        output_filename,  # Features Output FileName
        "../audio_output/" + audio_filepath  # Audio Filename
    ]
    
    try:
        # Run the Java command and set the current working directory to 'jAudio'
        subprocess.run(java_command, check=True, cwd="jAudio")
    except subprocess.CalledProcessError as e:
        print("Error executing Java command:", e)
    else:
        pass
        # print("Java command executed successfully")

        # Load XML file
        current_path = os.getcwd()
        xml_file = os.path.join(current_path, 'jAudio', f'{output_filename}FV.xml')
        # print("Current path:", current_path)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Create a dictionary to store the extracted data
        data = {'data_set_id': root.find('./data_set/data_set_id').text}
        #data = {"x":Decimal(0.0)}

        somInput = []
        # Iterate through feature elements and extract data

        finalFeatures = [feature for feature in coreFeatures.coreFeatures if feature not in coreFeatures.zero_features]
        for feature_name in coreFeatures.coreFeatures:
            feature_element = root.find(f"./data_set/feature[name='{feature_name}']")
            if feature_element is not None:
                values = [v.text for v in feature_element.findall('v')]
                if len(values) == 0:
                    print(feature_name)
                    somInput.append(0)
                    data[feature_name] = 0

                elif len(values) == 1:
                    if values[0] == "NaN":
                        print("zero parameter:",feature_name)
                        values[0] = 0
                    data[feature_name] = values[0]
                    somInput.append(float(Decimal(values[0])))
                    print(type(somInput[0]))
                else:
                    print(feature_name)
                    decimal_array = [float(Decimal(value)) for value in values]
                    # print(decimal_array)
                    # print(values)
                    average_decimal = sum(decimal_array) / len(decimal_array)
                    somInput.append(average_decimal)
                    data[feature_name] = average_decimal
            else:
                print("error:", feature_name)
        print(len(somInput))
                

   
        # Store the extracted data in a JSON file
        json_file = f'{output_folder}/{output_filename}_all_features.json'
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
        # print(f"Data extracted and saved to {json_file}")
    somInput = np.nan_to_num(somInput, nan=0.0)
    zeroCount=0
    for i in range(len(somInput)):
        if somInput[i] == 0.0:
            print("zero:", coreFeatures.coreFeatures[i])
            zeroCount+=1

    
    print("zeroCount:", zeroCount)
    print(somInput)

    print("LENGTH:", len(somInput))
    return somInput
