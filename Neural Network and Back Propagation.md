### Logistic regression
computation graph: ![](https://i.imgur.com/ZW8G5SG.png)
* input  X:    ($n_x$,m)
* weight w:    ($n_x$,1)
* bias   b：    constant
* output Y:    (1,m)

Forward propagation: 
1. $$Z=w^TX+b$$
2. $$A = Sigmoid(Z)$$

Cost function:
$$J(w,b) = 1/m\sum_{i=1}^m L(\hat{y}^i-y^i)
=1/m\sum_{i=1}^m [y^i\log \hat{y}^i+(1-y^i)\log (1-\hat{y}^i)]$$

Backward propagation:
1. $$da = \frac{\partial L}{\partial a} = \frac{-y}{a}+ \frac{1-y}{1-a} $$
2. $$dz = \frac{\partial L}{\partial a} \frac{\partial a}{\partial z} = (\frac{-y}{a}+ \frac{1-y}{1-a}) *a(1-a) = a-y$$
3. $$dw_1 = \frac{\partial L}{\partial z} \frac{\partial z}{\partial w_1} = dz * x_1 = x_1(a-y)$$
4. $$db = \frac{\partial L}{\partial z} \frac{\partial z}{\partial b} = dz * 1 = (a-y)$$

for m number of input the above equations can be changed to get the mean value of all derivatives 
![](https://i.imgur.com/uq2TqCd.png)
![](https://i.imgur.com/2VgyOSc.png)


### Neural Networks
it's best explained in this picture: 
![](https://i.imgur.com/7WIOXtt.png)

and the vectorization version can be written as followed: 
![](https://i.imgur.com/Hhqxkaw.png)

![](https://i.imgur.com/e91LxyH.png)
![](https://i.imgur.com/gXPxalp.png)
![](https://i.imgur.com/4MBZkGq.png)
![](https://i.imgur.com/kKdLwEO.png)

please notice in numpy we should have 
``` python 
np.zeros((3, 4))     # have a shape (3, 4)
np.random.randn(3,4) # have a shape (3, 4)

```
![](https://i.imgur.com/anjnuhD.png)
![](https://i.imgur.com/exelctj.png)
![](https://i.imgur.com/p1kQxDq.png)

He initialization: this is named for the first author of He et al., 2015. (If you have heard of "Xavier initialization", this is similar except Xavier initialization uses a scaling factor for the weights $W^{[l]}$ of `sqrt(1./layers_dims[l-1])` where He initialization would use `sqrt(2./layers_dims[l-1])`.)
``` python 
def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###
        
    return parameters
```
