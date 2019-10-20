import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt    # more on this later


data = pd.read_csv('ex2data1.txt', header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]

data.head()

# fuction of visualisation of data 

def visialisation_data():
        variables = {}
        
        with open("ex2data1.txt") as f:
            for line in f:
                X,Y,resultat = line.split(",")
                
                variables[X] = float(X)
                variables[Y] = float(Y)
                variables[resultat] = float(resultat)
                
                if variables[resultat]==0.0 :
                    plt.scatter(variables[X], variables[Y],c='coral') 
                else:
                    plt.scatter(variables[X], variables[Y],c='lightblue') 
                        
        plt.xlabel('Exam 1')
        plt.ylabel('Exam 2')
        axe=np.linspace(0,100,200000)
        plt.axis([20, 100, 20, 100])   
        plt.show()     
        f.close() ;
        
# the sigmoid function 

def sigmoid(z): 
       
        return  1/(1+np.exp(-z))

# CosFunction 

def costFunction(theta, X, y):
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta))) 
        + np.multiply((1-y), np.log(1 - sigmoid(X @ theta))))
    return J
        
# gradient function   

def gradient(theta, X, y):
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))
    
#    
(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X))
y = y[:, np.newaxis]


theta = np.zeros((n+1,1)) # intializing theta with all zeros
#J = costFunction(theta, X, y)
#print(J)        

temp = opt.fmin_tnc(func = costFunction, 
                    x0 = theta.flatten(),fprime = gradient, 
                    args = (X, y.flatten()))
#the output of above function is a tuple whose first element #contains the optimized values of theta
theta_optimized = temp[0] 
#J = costFunction(theta_optimized[:,np.newaxis], X, y)
 
 
#Separation of variable

plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
plot_y = -1/theta_optimized[2]*(theta_optimized[0] 
          + np.dot(theta_optimized[1],plot_x))  
mask = y.flatten() == 1
adm = plt.scatter(X[mask][:,1], X[mask][:,2])
not_adm = plt.scatter(X[~mask][:,1], X[~mask][:,2])
decision_boun = plt.plot(plot_x, plot_y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()    


def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y)
    print(acc * 100)
accuracy(X, y.flatten(), theta_optimized, 0.5)