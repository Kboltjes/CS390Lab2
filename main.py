import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import matplotlib.pyplot as plt

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# don't change these
ALL_DATASETS = ["mnist_d", "mnist_f", "cifar_10", "cifar_100_f", "cifar_100_c"]
ALL_ALGORITHMS = ["tf_net", "tf_conv"]


# Enable if you want to print a summary of the neural net layers
#DO_PRINT_NET_LAYERS = True
DO_PRINT_NET_LAYERS = False

# Set this to false if you just want to run the specified algorithm and dataset
# If you set this to true, it will iterate over every algorithm and dataset automatically and generate
# two pdfs that show accuracy for each dataset.
#GENERATE_BAR_GRAPHS = True
GENERATE_BAR_GRAPHS = False

#ALGORITHM = "guesser"
#ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

#DATASET = "mnist_d"
#DATASET = "mnist_f"
#DATASET = "cifar_10"
DATASET = "cifar_100_f"
#DATASET = "cifar_100_c"

#USE_PRETRAINED_WEIGHTS = True
USE_PRETRAINED_WEIGHTS = False

#TRAIN_USING_PRETRAINED_WEIGHTS = True
TRAIN_USING_PRETRAINED_WEIGHTS = False

# Just defining these variables globally. 
# To set their actual values, adjust them in updateDataSet()
NUM_CLASSES, IH, IW, IZ, IS, EPOCHS, BATCH_SIZE = (0,)*7
DROPOUT = False
DROP_RATE = 0.0
MODEL_FILENAME = ""

def updateDataSet():
    global NUM_CLASSES, IH, IW, IZ, IS, EPOCHS, BATCH_SIZE, DROPOUT, DROP_RATE, MODEL_FILENAME
    if DATASET == "mnist_d":
        NUM_CLASSES = 10
        IH = 28
        IW = 28
        IZ = 1
        IS = 784
        EPOCHS = 10
        DROPOUT = True
        DROP_RATE = 0.2
        BATCH_SIZE = 100
    elif DATASET == "mnist_f":
        NUM_CLASSES = 10
        IH = 28
        IW = 28
        IZ = 1
        IS = 784
        EPOCHS = 15
        DROPOUT = True
        DROP_RATE = 0.3
        BATCH_SIZE = 100
    elif DATASET == "cifar_10":
        NUM_CLASSES = 10
        IH = 32
        IW = 32
        IZ = 3
        IS = 3072
        EPOCHS = 10
        DROPOUT = True
        DROP_RATE = 0.2
        BATCH_SIZE = 100
    elif DATASET == "cifar_100_c":
        NUM_CLASSES = 20
        IH = 32
        IW = 32
        IZ = 3
        IS = 3072
        EPOCHS = 10
        DROPOUT = True
        DROP_RATE = 0.3
        BATCH_SIZE = 100
    elif DATASET == "cifar_100_f":
        NUM_CLASSES = 100
        IH = 32
        IW = 32
        IZ = 3
        IS = 3072
        EPOCHS = 10
        DROPOUT = True
        DROP_RATE = 0.4
        BATCH_SIZE = 100

    MODEL_FILENAME = DATASET + '_' + ALGORITHM


#=========================<Classes>================================

class NeuralNetwork_Keras():
    def __init__(self, didLoadFromFile=False, model=None):
        if didLoadFromFile:
            pass
        else:
            model = tf.keras.models.Sequential([ 
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.sigmoid), 
                tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.sigmoid)
                ])
            lossType = tf.keras.losses.BinaryCrossentropy()
            model.compile(optimizer='adam', loss=lossType, metrics=['accuracy'])
        self.model = model

    def train(self, xVals, yVals, epochs = 100):
        self.model.fit(xVals, yVals, epochs=epochs, batch_size=BATCH_SIZE)

    def predict(self, xVals):
        output = self.model.predict(xVals)
        return output

    def save(self, filename):
        self.model.save(filename)


class ConvNeuralNetwork_Keras():
    def __init__(self, dropout = False, dropRate = 0, didLoadFromFile=False, model=None):
        if didLoadFromFile:
            if DO_PRINT_NET_LAYERS:
                model.summary()
        else:
            model = keras.Sequential()
            lossType = keras.losses.categorical_crossentropy
            model = self.addLayers(model, dropout, dropRate)
            if DO_PRINT_NET_LAYERS:
                model.summary()
            model.compile(optimizer='adam', loss=lossType)
        self.model = model
    
    def addLayers(self, model, dropout, dropRate):
        inShape = (IH, IW, IZ)
        if DATASET == "mnist_d":
            model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "sigmoid", input_shape = inShape))
            model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"))
            model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
            if dropout:
                model.add(keras.layers.Dropout(dropRate))
            model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "sigmoid"))
            model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"))
            model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
            if dropout:
                model.add(keras.layers.Dropout(dropRate))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(128, activation = "relu"))
            model.add(keras.layers.Dense(NUM_CLASSES, activation = "softmax"))
        elif DATASET == "mnist_f":
            model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "sigmoid", input_shape = inShape))
            model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"))
            model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
            if dropout:
                model.add(keras.layers.Dropout(dropRate))
            model.add(keras.layers.Conv2D(48, kernel_size = (3, 3), activation = "relu"))
            model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"))
            if dropout:
                model.add(keras.layers.Dropout(dropRate))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(128, activation = "relu"))
            model.add(keras.layers.Dense(64, activation = "relu"))
            model.add(keras.layers.Dense(NUM_CLASSES, activation = "softmax"))
        elif DATASET == "cifar_10":
            model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "sigmoid", padding="same", input_shape = inShape))
            model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu", padding="same"))
            model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
            if dropout:
                model.add(keras.layers.Dropout(dropRate))
            model.add(keras.layers.Conv2D(48, kernel_size = (3, 3), activation = "relu", padding="same"))
            model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu", padding="same"))
            model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
            if dropout:
                model.add(keras.layers.Dropout(dropRate))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(128, activation = "relu"))
            model.add(keras.layers.Dense(128, activation = "relu"))
            model.add(keras.layers.Dense(NUM_CLASSES, activation = "softmax"))
        elif DATASET == "cifar_100_c":
            model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "sigmoid", padding="same", input_shape = inShape))
            model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu", padding="same"))
            model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
            if dropout:
                model.add(keras.layers.Dropout(dropRate))
            model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "relu", padding="same"))
            model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu", padding="same"))
            model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
            if dropout:
                model.add(keras.layers.Dropout(dropRate))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(128, activation = "relu"))
            model.add(keras.layers.Dense(64, activation = "relu"))
            model.add(keras.layers.Dense(NUM_CLASSES, activation = "softmax"))
        elif DATASET == "cifar_100_f":
            model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "sigmoid", padding="same", input_shape = inShape))
            model.add(keras.layers.Conv2D(128, kernel_size = (3, 3), activation = "relu", padding="same"))
            model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
            if dropout:
                model.add(keras.layers.Dropout(dropRate))
            model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu", padding="same"))
            model.add(keras.layers.Conv2D(128, kernel_size = (3, 3), activation = "relu", padding="same"))
            model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
            if dropout:
                model.add(keras.layers.Dropout(dropRate))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(512, activation = "relu"))
            model.add(keras.layers.Dense(256, activation = "relu"))
            model.add(keras.layers.Dense(128, activation = "relu"))
            model.add(keras.layers.Dense(NUM_CLASSES, activation = "softmax"))
        return model

    def train(self, xVals, yVals, epochs = 100):
        self.model.fit(xVals, yVals, epochs=epochs, batch_size=BATCH_SIZE)

    def predict(self, xVals):
        output = self.model.predict(xVals)
        return output

    def save(self, filename):
        self.model.save(filename)

#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 6, doBuildNewNet=True, model=None):
    if doBuildNewNet:
        model = NeuralNetwork_Keras()
    model.train(x, y, eps)
    return model


def buildTFConvNet(x, y, doBuildNewNet=True, model=None):
    if doBuildNewNet:
        model = ConvNeuralNetwork_Keras(dropout=DROPOUT, dropRate=DROP_RATE)
    model.train(x, y, EPOCHS)
    return model

#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        cifar10 = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    elif DATASET == "cifar_100_f":
        cifar100 = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar100.load_data(label_mode="fine")
    elif DATASET == "cifar_100_c":
        cifar100 = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar100.load_data(label_mode="coarse")
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data, doBuildNewNet=True, model=None):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        if doBuildNewNet:
            print("Building and training TF_NN.")
        else:
            print("Training TF_NN using pre-existing weights.")
        return buildTFNeuralNet(xTrain, yTrain, doBuildNewNet=doBuildNewNet, model=model)
    elif ALGORITHM == "tf_conv":
        if doBuildNewNet:
            print("Building and training TF_CNN.")
        else:
            print("Training TF_CNN using pre-existing weights.")
        return buildTFConvNet(xTrain, yTrain, doBuildNewNet=doBuildNewNet, model=model)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print(f"Testing TF_NN on {DATASET}.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print(f"Testing TF_CNN on {DATASET}.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    return accuracy


def loadModel(filename):
    model = keras.models.load_model(filename)
    if ALGORITHM == "tf_conv":
        return ConvNeuralNetwork_Keras(didLoadFromFile=True, model=model)
    elif ALGORITHM == "tf_net":
        return NeuralNetwork_Keras(didLoadFromFile=True, model=model)


#=========================<Main>================================================

def main():
    # Update global variables
    updateDataSet()

    raw = getRawData()
    data = preprocessData(raw)
    if USE_PRETRAINED_WEIGHTS and os.path.exists(MODEL_FILENAME):
        print(f"Loading trained model: {MODEL_FILENAME}")
        model = loadModel(MODEL_FILENAME)
        if TRAIN_USING_PRETRAINED_WEIGHTS:
            print("Training using pretrained weights")
            model = trainModel(data[0], doBuildNewNet=False, model=model)
    else:
        model = trainModel(data[0])

    if USE_PRETRAINED_WEIGHTS:
        print(f"Saving Trained Model: {MODEL_FILENAME}")
        model.save(MODEL_FILENAME)

    preds = runModel(data[1][0], model)
    return evalResults(data[1], preds)



if __name__ == '__main__':
    if GENERATE_BAR_GRAPHS:
        # generate the plots by iterating over all algorithms and all datasets
        for algorithm in ALL_ALGORITHMS:
            ALGORITHM = algorithm

            algoName = "CNN" if ALGORITHM == "tf_conv" else "ANN"
                
            accuracies = []
            for dataset in ALL_DATASETS:
                DATASET = dataset
                accuracies.append(main())

            # make a plot and save it as a png
            plt.bar(ALL_DATASETS, accuracies)
            plt.xlabel("Dataset")
            plt.ylabel("Accuracy")
            plt.title(f"{algoName} Accuracies")
            plt.savefig(f"{algoName}_Accuracy_Plot.pdf")
        pass
    else:
        # just run the network normally
        main()
