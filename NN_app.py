from execution import *
X = None
Y = None
layers_dims = None
learning_rate = None
num_iterations = None
print_cost = False
""""
REMOVE THE COLONS AND LOAD YOUR DATASET
"""
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
X, Y = load_planar_dataset()

noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p



datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

dataset = "blobs"
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

if dataset == "blobs":
    Y = Y%2
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)


layer_dims = [2,10,1]
num_iterations = 10000
learning_rate = 1.2
print(X,Y,layer_dims)
params, costs = two_layer_model(X,Y,layer_dims,learning_rate,num_iterations,True)
p = predict(X,Y,params)
