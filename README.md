# Predicting Permeability of Porous Media Using Artificial Neural Networks
This repository contains code for predicting the permeability of porous media using artificial neural networks (ANN).

##Overview
The goal of this project is to develop a predictive model for permeability of porous media using ANN. Permeability is a critical parameter in the oil and gas industry, and it is expensive and time-consuming to measure experimentally. Thus, developing a predictive model for permeability using easily measurable properties such as porosity, specific surface area, and Kozeny-Carman constant can save time and resources.

## Dataset
The dataset used for this project was generated synthetically. Random values for porosity, specific surface area, and Kozeny-Carman constant were generated using the NumPy library. Permeability was then calculated using the Kozeny-Carman equation. The data was normalized using the MinMaxScaler function from the Scikit-learn library.

## Artificial Neural Networks
Artificial neural networks (ANN) are a type of machine learning model that simulate the behavior of the human brain. They consist of layers of interconnected nodes that process and transmit information. ANN can be trained to recognize patterns and relationships in data, and to make predictions based on that data.

## Model Architecture
The ANN model used in this project consists of three layers: an input layer with three nodes corresponding to the three input variables (porosity, specific surface area, and Kozeny-Carman constant), a hidden layer with 16 nodes, and another hidden layer with eight nodes. The output layer has one node, which predicts the permeability. The activation function used in the hidden layers is the rectified linear unit (ReLU) function, and the output layer has a linear activation function.

## Training the Model
The model was trained using the Adam optimizer and mean squared error (MSE) as the loss function. The data was split into training, validation, and testing sets using the train_test_split function from Scikit-learn. The model was trained for 100 epochs with a batch size of 32. The performance of the model was evaluated using the test set, and the mean squared error (MSE), root mean squared error (RMSE), and R-squared (R2) values were calculated.

## Results
The model achieved a test MSE of 0.0003, test RMSE of 0.018, and test R2 of 0.995. These results indicate that the model is able to accurately predict permeability using porosity, specific surface area, and Kozeny-Carman constant as input variables.

## Usage
To run the code, make sure you have the following libraries installed: NumPy, Scikit-learn, Keras, and Matplotlib. You can run the code in Jupyter Notebook or any Python IDE.

