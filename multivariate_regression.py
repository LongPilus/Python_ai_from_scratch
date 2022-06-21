import numpy as np;from matplotlib import pyplot as plt

def multivar_regression(X: np.matrix,Y: np.array):

    m = X.shape[0];n = X.shape[1] # m, n
    theta = np.ones(n+1) # Theta
    X = np.c_[np.ones(m),X]# Add X_0 to matrix X

    # Feature scaling
    def feature_scaling(X,Y):
        m = X.shape[0];n = X.shape[1]-1
        # Sclaing X
        X_scale = np.ones(n+1)
        for col in range(1,n+1):
            X_scale[col]=max(X[:,col])
            X[:,col]=X[:,col]/max(X[:,col])
        # Scaling Y
        Y_scale = max(Y)
        Y = Y/max(Y)

        return X, Y, X_scale, Y_scale

    X, Y, X_scale, Y_scale = feature_scaling(X,Y)

    # Cost function
    def cost_function(X: np.matrix,Y: np.array,theta: np.array):
        alpha = 0.001 # Alpha
        m = X.shape[0];n = X.shape[1]-1
        hypothesis = np.sum(np.tile(theta,(m,1))*X,1)

        hypothesis = np.sum(np.tile(theta,(m,1))*X,1)
        J = 1/m/2*np.sum((hypothesis-Y)**2)
        theta = theta-(alpha/m*sum(np.tile(np.sum(np.tile(theta,(m,1))*X,1)-Y,(n+1,1)).T*X,1))
        return J, theta

    iters = range(1000)
    J_iters = np.ones(len(iters))
    for i in iters:
        J, theta = cost_function(X,Y,theta)
        J_iters[i]=J
#    plt.plot(iters,J_iters)
    return theta

x = np.array([[2104,5,1,45],[1416,3,2,40],[1534,3,2,30],[852,2,1,36]])
y = np.array([460,232,315,178])

print(multivar_regression(x,y))
#plt.show()
