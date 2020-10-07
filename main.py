import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from numpy import loadtxt
from tensorflow.keras.models import load_model
import os.path


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ALGORITHM = "guesser"
ALGORITHM = "tf_net"
# ALGORITHM = "tf_conv"

DATASET = "mnist_d"
# DATASET = "mnist_f"
# DATASET = "cifar_10"
# DATASET = "cifar_100_f"
# DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 20
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
    
    
    
#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 15):
    #TODO: Implement a standard ANN here.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512,activation = tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CLASSES,activation = tf.nn.softmax)
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x, np.argmax(y, axis=1), batch_size=128, epochs=eps, verbose = 1)
    return model


def buildTFConvNet(x, y, eps = 25, dropout = True, dropRate = 0.2):
    #TODO: Implement a CNN here. dropout option is required.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation = tf.nn.relu, padding = "same", input_shape=(IH, IW, IZ)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation = tf.nn.relu, padding = "same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(dropRate),
        tf.keras.layers.Conv2D(64, (3, 3), activation = tf.nn.relu, padding = "same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation = tf.nn.relu, padding = "same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(dropRate),
        tf.keras.layers.Conv2D(128, (3, 3), activation = tf.nn.relu, padding = "same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation = tf.nn.relu, padding = "same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(dropRate),
        tf.keras.layers.Conv2D(256, (3, 3), activation = tf.nn.relu, padding = "same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation = tf.nn.relu, padding = "same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(dropRate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation = tf.nn.relu),
        tf.keras.layers.Dropout(dropRate),
        tf.keras.layers.Dense(NUM_CLASSES,activation = tf.nn.softmax)
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x, np.argmax(y, axis=1), batch_size=128, epochs=eps, verbose = 1)
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
    xTrain = xTrain / 255
    xTest = xTest / 255
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
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
    print("Classifier f1 Score: %f%%" % (f1_score(yTest, preds, average='micro') * 100))
    print("Confusion Matrix:")
    print(confusion_matrix(np.argmax(yTest, axis=1), np.argmax(preds, axis=1)))
    print()
    
    

#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model_name = ALGORITHM + ">" + DATASET + ".h5"
    
    if os.path.isfile('./' + model_name):
        # load model
        print("Model: " + model_name)
        load = load_model(model_name)
        preds = runModel(data[1][0], load)
        evalResults(data[1], preds)
        
    else:
        model = trainModel(data[0])
        preds = runModel(data[1][0], model)
        evalResults(data[1], preds)
        # save model
        model.save(model_name)
        print("Saved model to disk")
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    %matplotlib inline
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    x=['mnist_d','mnist_f','cifar_10','cifar_100_c','cifar_100_f']
    y = [98.44,88.47,43.87,28.56,18.74]
    bars = plt.bar(x, height=y, width=.4)
    plt.bar(x,y)
    plt.xlabel('datasets')
    plt.ylabel("accuracy (%)")
    plt.title('ANN Accuracy Plot')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval)
    plt.savefig("ANN_Accuracy_Plot.pdf")
    plt.show()

    x=['mnist_d','mnist_f','cifar_10','cifar_100_c','cifar_100_f']
    y = [99.29,93.49,83.27,64.55,53.21]
    bars = plt.bar(x, height=y, width=.4)
    plt.bar(x,y)
    plt.xlabel('datasets')
    plt.ylabel("accuracy (%)")
    plt.title('CNN Accuracy Plot')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval)
    plt.savefig("CNN_Accuracy_Plot.pdf")
    plt.show()

    


if __name__ == '__main__':
    main()
