import som.som_class
from Feature_Extractor import extract_features
from som.som_class import SOM
import matplotlib.pyplot as plt

s = SOM()

# Mapping dictionary for key transformation
key_mapping = {1: 'Karuna', 2: 'Shanta', 3: 'Shringar', 4: 'Veera'}

X = extract_features(filename="karuna__9.wav")

print("number of extracted features:", len(X))
print("features:", X)

rasaDob = s.predict(X)

print("rasaDob:", rasaDob)



# Transform keys in the dictionary
rasaDob_transformed = {key_mapping.get(key, key): value for key, value in rasaDob.items()}

# Your transformed dictionary
print("rasaDob_transformed:", rasaDob_transformed)

# Extract keys and values from the transformed dictionary
keys = list(rasaDob_transformed.keys())[:-1]  # Remove the last value
values = list(rasaDob_transformed.values())[:-1]  # Remove the last value

# Define colors for each x label
colors = ['blue', 'green', 'red', 'purple']

# Plot the scatter plot with different colors for each x label
scatter = plt.scatter(keys, values, c=colors, marker='o', s=100, edgecolors='black')

# Add labels and title
plt.xlabel('Rasa')
plt.ylabel('Degree of Belonging')
plt.title('DOB Values for each Rasa. Song: Shringar')

# Add legend
legend_labels = ['Karuna', 'Shanta', 'Shringar', 'Veera', f'Predicted: {key_mapping[list(rasaDob_transformed.values())[4]]}']
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10))  # Predicted
plt.legend(legend_handles, legend_labels, title='Rasas', loc='upper right')

# Show the plot
plt.show()
