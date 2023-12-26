import numpy as np
from numpy.ma.core import ceil
from scipy.spatial import distance #distance calculation
from sklearn.preprocessing import MinMaxScaler #normalisation
import matplotlib.pyplot as plt
from collections import Counter

# Helper functions

# Data Normalisation
def minmax_scaler(data):
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(data)
  return scaled

# Euclidean distance
def e_distance(x,y):
  return distance.euclidean(x,y)

# Manhattan distance
def m_distance(x,y):
  return distance.cityblock(x,y)


## Degree of Belonging
def winning_neuron(data, t, som, num_rows, num_cols):
    winner = [0, 0]
    shortest_distance = np.sqrt(data.shape[1])  # initialize with max distance
    input_data = data[t]
    for row in range(num_rows):
        for col in range(num_cols):
            distance = e_distance(som[row][col], input_data)
            if distance < shortest_distance:
                shortest_distance = distance
                winner = [row, col]
    
    return winner, shortest_distance

def winning_neuron_per_rasa(data, t, som, num_rows, num_cols, label_map, rasa):
    winner = [0, 0]
    shortest_distance = np.sqrt(data.shape[1])  # initialize with max distance
    input_data = data[t]
    for row in range(num_rows):
        for col in range(num_cols):
            if label_map[row][col] == rasa:
                distance = e_distance(som[row][col], input_data)
                if distance < shortest_distance:
                    shortest_distance = distance
                    winner = [row, col]
    
    return winner, shortest_distance

def get_shortest_distances_per_rasa(data, t, som, num_rows, num_cols, num_classes, label_map):
    df = {}
    for rasa in range(1,num_classes+1):
        winner, shortest_distance = winning_neuron_per_rasa(data, t, som, num_rows, num_cols, label_map, rasa)
        df[rasa] = shortest_distance
    winner, shortest_distance = winning_neuron(data, t, som, num_rows, num_cols)
    row = winner[0]
    col = winner[1]
    df["Predicted"] = label_map[row][col]
    return label_map[row][col], df


# Learning rate and neighbourhood range calculation
def decay(step, max_steps,max_learning_rate,max_m_dsitance):
  coefficient = 1.0 - (np.float64(step)/max_steps)
  learning_rate = coefficient*max_learning_rate
  neighbourhood_range = ceil(coefficient * max_m_dsitance)
  return learning_rate, neighbourhood_range