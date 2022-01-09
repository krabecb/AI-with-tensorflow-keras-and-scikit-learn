#pandas is an open source data analysis and manipulation tool
#pandas has tools for reading and writing data between in-memory
#data structures and different formats, in this case, CSV
import pandas as pd

dataset = pd.read_csv('./cancer.csv')

#set up x and y attributes
#ai will map the correlations between these two features and that allows
#ai to predict whether a tumor is m or b
#x: all columns except diagnosis column
x = dataset.drop(columns=['diagnosis(1=m, 0=b)'])
#y: diagnosis column
y = dataset['diagnosis(1=m, 0=b)']

#next, split up data between a training set and a testing set
#import in ai - often times algorithms are overfitting(modeling error in statistics)
#means that alorithm does really well on data its seen but new data tends to fall apart
#to mitigate this, we're going to set part of our dataset aside to be tested later with the algorithm
#the algorithm will be given data that hasn't been seen before and then we'll use how well it does on that data
#to evaluate
#overall focus: not how well an algorithm does on an entire dataset bu the general problem
#which is cancer diagnosis

#sklearn - machine learning library
#Used to split dataset between training set and testing set
from sklearn.model_selection import train_test_split

#test_size=0.2: 20% of our data is going to be in the testing set
#fairly common to see 80%-20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#build ai and nueral network - a popular form of ai used in most problems and adapts to it
import tensorflow as tf

model= tf.keras.models.Sequential()

#add layers to module
#neural network: input layer is the x values of the cancer dataset -> hidden layer -> output value is if tumor is m or b
#Dense: standard vanilla/default neurons in keras
#256: make neural network bigger than it has to be - to see how powerful we can get with this dataset
#the first model.add is the input layer of the neural network, so there is a notion of input_shape, all of the x features
#activation function: all the values from the neural network and plotting them between 0 and 1 helpful to reduce model complexity and 
#make model more accurate
model.add(tf.keras.layers.Dense(256, input_shape=[None, x_train.shape[0], x_train.shape[1]], activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
#1: our final value is one single value between 0 and 1 for diagnosis
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#compile model
#optimizer='adam': how the machine learning algorithm is being optimized how the neurons/weights of the algorithm
#are being fine-tuned to fit the data
#loss: Because we're doing binary classification, use a metric called binary_crossentropy - good for categorical stuff
#metrics: Correctly classify as many tumors as possible
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#epochs=1000: how many times the algorithm is iterating over the same data
#1000 is a little overkill, but better to be safe than sorry
model.fit(x_train, y_train, epochs=1000)

#comparing what the model thinks the y_test should be vs what y_test actually is
model.evaluate(x_test, y_test)