import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from numpy.ma.core import ceil
from scipy.spatial import distance #distance calculation
from sklearn.preprocessing import MinMaxScaler #normalisation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #scoring
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from collections import Counter
import random
from som.som_helper import *
import pickle
import joblib

class SOM:
    def __init__(self):
        self.num_rows = 12     
        self.num_cols = 12
        self.max_m_distance = 4
        self.max_learning_rate = 0.4
        self.max_steps = int(3*10e3)
        self.num_classes = 4
        self.train_x, self.test_x, self.train_y, self.test_y, self.X_std, self.X, self.y = self.loadData()

        #print(self.train_x.shape)
        #self.som = self.train(self.X_std, self.train_x, self.test_x, self.train_y, self.test_y)
        #self.label_map = self.labelMap(self.som, self.X_std, self.y, self.train_x, self.train_y, self.num_rows, self.num_cols)

        self.test(self.train_x, self.test_x, self.test_y, self.X_std)
        

    def loadData(self):
        ### MODIFY THE NAME OF YOUR CSV HERE
        df_data = pd.read_csv('./som/y.csv')
        y = df_data['Navarasa']
        columns_to_drop = ['data_set_id', 'Cluster', 'Srno', 'Belonging to C-0', 'Belonging to C-1', 'Navarasa', 'sno']
        extra_features_to_drop = ['Derivative of Method of Moments Overall Average', 'Derivative of Running Mean of Method of Moments Overall Average', 'Log of ConstantQ Overall Average']
        df_data = df_data.drop(columns=columns_to_drop, axis=1)
        df_data = df_data.drop(columns=extra_features_to_drop, axis=1)

        df_data = df_data.dropna(axis=1)  # Drop rows with NaN in any column
        df_data = df_data.loc[:, (df_data != 0).any(axis=0)]  # Drop rows with zero in any column
        column_list = df_data.columns.values.tolist()
        ##### MODIFY HERE BASED ON YOUR DATA COLOUMNS AND LABEL COLUMNS
        #Extract x and y from the dataframe
        X = df_data.iloc[:,0:253].values
        df_subset = df_data[column_list[:253]]
        file_path = './som/loadingDataCOlumns.txt'
        numbers = df_subset

        # Open the file in write mode
        with open(file_path, 'w') as file:
            # Write each number to the file
            for number in numbers:
                file.write(str(number) + '\n')
        
        y = y.values

        #Standardize data
        Scaler = StandardScaler().fit(X)
        X_std = Scaler.transform(X)
        joblib.dump(Scaler, './som/scaler.pkl')

        minmaxScaler = MinMaxScaler().fit(X_std)
        X_norm = minmaxScaler.transform(X_std)
        # X_norm = X_std
        joblib.dump(minmaxScaler, './som/minmaxScaler.pkl')

        # print(X_std.shape)

        # #applying PCA 
        # model = PCA(n_components=60).fit(X_std)
        # X_pc = model.transform(X_std)

        # # number of components
        # n_pcs= model.components_.shape[0]

        train_x, test_x, train_y, test_y = train_test_split(X_std, y, test_size=0.2, random_state=42)
        return train_x, test_x, train_y, test_y, X_std, X, y
    
    def train(self, X_std, train_x, test_x, train_y, test_y):
        minmaxScaler = MinMaxScaler().fit(X_std)
        X_norm = minmaxScaler.transform(X_std)
        # X_norm = X_std
        joblib.dump(minmaxScaler, './som/minmaxScaler.pkl')

        # print(X_norm.shape)

        # initialising self-organising map
        num_dims = X_norm.shape[1] # numnber of dimensions in the input data
        np.random.seed(40)
        # som = np.random.random_sample(size=(num_rows, num_cols, num_dims)) # map construction
        som = np.zeros((self.num_rows, self.num_cols, num_dims))
        # start training iterations
        for step in range(self.max_steps):
            activation_list = random.sample(range(0,self.num_rows+self.num_cols),int((self.num_rows+self.num_cols)/1.3))
            # print(activation_list)
            if (step+1) % 1000 == 0:
                print("Iteration: ", step+1) # print out the current iteration for every 1k
            learning_rate, neighbourhood_range = decay(step, self.max_steps,self.max_learning_rate,self.max_m_distance)

            t = np.random.randint(0,high=X_norm.shape[0]) # random index of traing data
            winner,shortest_distance = winning_neuron(X_norm, t, som, self.num_rows, self.num_cols)
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                # if (row+col) in activation_list:

                #   # print("hi")
                #   continue

                    if m_distance([row,col],winner) <= neighbourhood_range:
                        som[row][col] += learning_rate*(X_norm[t]-som[row][col]) # update neighbour's weight

        print("SOM training completed")
        with open('./som/som_model.pkl', 'wb') as f:
            pickle.dump(som, f, protocol=0)

        return som

    def labelMap(self, som, X_std, y, train_x, train_y, num_rows, num_cols):
        # collecting labels
        minmaxScaler = MinMaxScaler().fit(X_std)
        X_norm = minmaxScaler.transform(X_std)
        # X_norm = X_std
        label_data = y
        map = np.empty(shape=(num_rows, num_cols), dtype=object)

        for row in range(num_rows):
            for col in range(num_cols):
                map[row][col] = [] # empty list to store the label

        for t in range(X_norm.shape[0]):
            if (t+1) % 1000 == 0:
                print("sample data: ", t+1)
            winner, shortest_distance = winning_neuron(X_norm, t, som, num_rows, num_cols)
            map[winner[0]][winner[1]].append(label_data[t]) # label of winning neuron

        # construct label map
        label_map = np.zeros(shape=(num_rows, num_cols),dtype=np.int64)
        for row in range(num_rows):
            for col in range(num_cols):
                label_list = map[row][col]
                if len(label_list)==0:
                    label = 2
                else:
                    label = max(label_list, key=label_list.count)
                label_map[row][col] = label

        with open('./som/labelMap.pkl', 'wb') as f:
            pickle.dump(label_map, f, protocol=0)

        return label_map

    def test(self, train_x,test_x, test_y, X_std):

        #X_std = StandardScaler().fit_transform(X)
        #test_x_norm = minmax_scaler(test_x)

        with open('./som/som_model.pkl', 'rb') as f:
            som = pickle.load(f, encoding='ASCII')
        with open('./som/labelMap.pkl', 'rb') as f:
            label_map = pickle.load(f, encoding='ASCII')
        data = minmax_scaler(train_x) # normalisation
        minmaxScaler = MinMaxScaler().fit(X_std)
        X_norm = minmaxScaler.transform(X_std)
        # X_norm = X_std

        file_path = './som/testing.txt'
        numbers = X_norm[34]

        # Open the file in write mode
        with open(file_path, 'w') as file:
            # Write each number to the file
            for number in numbers:
                file.write(str(number) + '\n')
        winner_labels = []
        df = []
        num_classes = 4
        rasas = ["KARUNA", "SHANTA", "SHRINGAR", "VEERA"]
        for t in range(X_norm.shape[0]):
            winner, rasa_df = get_shortest_distances_per_rasa(X_norm, t, som, self.num_rows, self.num_cols, num_classes, label_map)
            # print(winner)
            winner_labels.append(winner)
            rasa_df["REAL"] = self.y[t]
            df.append(rasa_df)


        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(df)

        # Save the DataFrame to a CSV file
        df.to_csv("./som/final_classes_per_rasa.csv", index=False)

        count=0
        for i in range(len(X_norm)):
            if self.y[i] == winner_labels[i]:
                count+=1
        #print(count, "/", len(X_norm))
        print("Accuracy: ",accuracy_score(self.y, np.array(winner_labels)))

    def predict(self, X):
        #print(X)
        #test_x_norm = minmax_scaler(test_x)

        with open('./som/som_model.pkl', 'rb') as f:
            som = pickle.load(f, encoding='ASCII')
        with open('./som/labelMap.pkl', 'rb') as f:
            label_map = pickle.load(f, encoding='ASCII')

        loaded_scaler = joblib.load('./som/scaler.pkl')
        minmaxScaler = joblib.load('./som/minmaxScaler.pkl')
        # print(X)
        X = X.reshape(1, -1)
        X = loaded_scaler.transform(X)
        X = minmaxScaler.transform(X)
        # print("standard X:", X)

        X_all = np.vstack((self.train_x, X))
        # X_all=X_all[1:]

        X_std = StandardScaler().fit_transform(X_all)
        #print("STANDARD:", X_std)
        test_x_norm = minmax_scaler(X)

        file_path = './som/predicting.txt'
        numbers = X[len(X)-1]

        # Open the file in write mode
        with open(file_path, 'w') as file:
            # Write each number to the file
            for number in numbers:
                file.write(str(number) + '\n')

        # print("standardized final x:", X)
        lables = []
        ### MODIY NUMBER LABELS ACCORDING TO THE NUMBER OF CLASSES IN YOUR CSV
        num_labels = 4
        # d,_,_ = get_bmu(test_x_norm[len(test_x_norm)-1],som,num_labels,label_map)
        winner, rasa_df = get_shortest_distances_per_rasa(X, len(X)-1, som, self.num_rows, self.num_cols, self.num_classes, label_map)
        #print(rasa_df)
        return rasa_df