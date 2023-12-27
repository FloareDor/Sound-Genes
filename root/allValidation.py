import som.som_class
from Feature_Extractor import extract_features
from som.som_class import SOM
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

s = SOM()

df_data = pd.read_csv('y.csv')

audioFilenames = df_data['data_set_id']
y = df_data['Navarasa']

count = 0
df = []
for t in range(len(audioFilenames)):
		# data = minmax_scaler(train_x) # normalisation
	# X_norm = minmax_scaler(X_std)
	winner_labels = []

	num_classes = 4
	rasas = ["KARUNA", "SHANTA", "SHRINGAR", "VEERA"]
	X = extract_features(filename=audioFilenames[t])
	rasaDob = s.predict(X)
	winner_labels.append(rasaDob["Predicted"])
	rasaDob["REAL"] = y[t]
	if rasaDob["Predicted"] == y[t]:
		count+=1
	df.append(rasaDob)
	print(rasaDob)


# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(df)

# Save the DataFrame to a CSV file
df.to_csv("validation.csv", index=False)

print("accuracy:", count/(len(audioFilenames)), count, "/", len(audioFilenames))