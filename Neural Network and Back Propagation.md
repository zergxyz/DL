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
### Loss function
Multiclass SVM: The SVM loss is set up so that the SVM “wants” the correct class for each image to a have a score higher than the incorrect classes by some fixed margin Δ. Notice that it’s sometimes helpful to anthropomorphise the loss functions as we did above: The SVM “wants” a certain outcome in the sense that the outcome would yield a lower loss (which is good).
$$L_i = \sum_{j\neq y_i}\max(0, s_j - s_{y_i} + \Delta)，$$
这里 $\Delta$通常设置为1.0 

SVM gradients: 
对一个sample $x_i$，svm的loss为：

$$
\begin{aligned}
L_i = & \sum_{j \neq y_i}^C \max\left( 0, w_j x_i - w_{y_i} x_i + \Delta \right) \newline
= & \max\left( 0, w_0 x_i - w_{y_i} x_i + \Delta \right) + ... + \max\left( 0, w_j x_j - w_{y_i} x_i + \Delta \right) + ...
\end{aligned}
$$

$L_i$ 对 $w_j$ 求导：

$$
\mathrm{d}w_j =  \frac{\partial L_i}{\partial w_j} = 0 + 0 + ... +
 \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \cdot x_i
$$

$L_i$ 对 $w_{y_i}$ 求导：

$$
\begin{aligned}
\mathrm{d}w_{y_i} =& \frac{\partial L_i}{\partial w_{y_i}} =
\mathbb{1} \left( w_0 x_i - w_{y_i} x_i + \Delta > 0\right) \cdot (-x_i) +
 ... + \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \cdot (-x_i) + ... \newline
 =& - \left(  \sum_{j \neq y_i}^C  \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \right) \cdot x_i
 \end{aligned}
$$
``` python 
def svm_loss_vectorized(W, X, y, reg):
    loss, dW = None, None
    N = X.shape[0]
    C = W.shape[1]
    dW = np.zeros_like(W)
    
    scores = X.dot(W)
    correct_score = scores[np.arange(N), y]
    scores = np.maximum(0, scores - correct_score[:, np.newaxis] + 1.0)
    scores[np.arange(N), y] = 0
    dScore = (scores > 0).astype(np.float)
    dScore[np.arange(N), y] = -np.sum(dScore, axis=1)
    
    dW = X.T.dot(dScore)
    loss = np.sum(scores) / N + reg * np.sum(W * W)
    dW = dW / N + 2 * reg * W
    return loss, dW
```

Softmax: cross-entropy loss that has the form 
$$L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \hspace{0.5in} \text{or equivalently} \hspace{0.5in} L_i = -f_{y_i} + \log\sum_j e^{f_j}$$
Softmax gradients: 
还是要stage到score级别，然后再用 $\mathrm{d} W = X.T.dot(\mathrm{d} Score)$来计算 $\mathrm{d} W$，这样可以在推导的时候不用考虑如何计算对两个矩阵相乘。
$$
L_i = - \log \left( \ p_{y_i} \right) = -\log \left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right )
$$

$L_i$ 对任意 $s_k$ 求导：
$$
\begin{aligned}
\mathrm{d} s_k =& \frac{\partial L_i}{\partial s_k} = - \frac{\partial}{\partial s_k} \left( \log \left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right ) \right) \newline
=& - \frac{\sum_j e^{s_j}}{e^{s_{y_i}}} \cdot \frac{\left( {e^{s_{y_i}}}\right)^{'} \cdot {\sum_j e^{s_j}} - {e^{s_{y_i}}} \cdot \left( {\sum_j e^{s_j}} \right)^{'}}{\left( {\sum_j e^{s_j}}\right)^2} \newline
=&\frac{\frac{\partial}{\partial s_k}\left( {\sum_j e^{s_j}} \right)}{{\sum_j e^{s_j}}} - \frac{ \frac{\partial }{\partial s_k} \left({e^{s_{y_i}}} \right)}{{e^{s_{y_i}}}} \newline
=&\frac{\frac{\partial}{\partial s_k}\left( e^{s_0} + e^{s_1} + e^{s_{y_i}} + ... \right)}{{\sum_j e^{s_j}}} - \frac{ \frac{\partial }{\partial s_k} \left({e^{s_{y_i}}} \right)}{{e^{s_{y_i}}}}
\end{aligned}
$$
当 $y_i = k$时：
$$
\mathrm{d} s_k = \frac{{e^{s_{y_i}}}}{{\sum_j e^{s_j}}} - 1
$$
当 $y_i \neq k$时：
$$
\mathrm{d} s_k = \frac{{e^{s_k}}}{{\sum_j e^{s_j}}}
$$
综上，
$$
\mathrm{d} s_k = \frac{{e^{s_k}}}{{\sum_j e^{s_j}}} - \mathbb{1} \left( y_i = k \right)
$$
``` python 
def softmax_loss_vectorized(W, X, y, reg):
    loss, dW = 0.0, None
    N = X.shape[0]
    C = W.shape[1]
    
    scores = X.dot(W)
    Dscores = np.zeros_like(scores)
    stable_scores = scores - np.max(scores, axis=1, keepdims=True)
    correct_score = stable_scores[np.arange(N), y]
    
    loss = -np.sum(np.log(np.exp(correct_score) / np.sum(np.exp(stable_scores), axis=1)))
    loss = loss/N + reg*np.sum(W*W)
    
    Dscores = np.exp(stable_scores) / np.sum(np.exp(stable_scores), axis=1, keepdims=True)
    Dscores[np.arange(N), y] -= 1
    Dscores = Dscores / N

    dW = X.T.dot(Dscores)
    dW = dW + 2*reg*W
    return loss, dW
```
### Back propagation for 2 layer NN
based on Andrew's advice we should use the following architecture: 
* input X should have size (C, N)
* W1 should have shape (nW1, C)
Basically we put size of data in the column position and put the feature dimension in the row position for X. 
Backprop和chain rule，就是用求解微分时的链式法则，将复杂算式的微分计算，一步步分解成小的node，然后用这些基本的node层层叠加，最后得到微分结果。通常做法是先画出computation graph，然后再stage by stage的计算grads，基本的公式是：
>                      down_diff = local_diff * up_diff
其中up_diff是从上一层block传递下来的，local_diff要通过计算得到，并且和输入值有关，两者相乘传递给下一层的block。道理很简单，但是具体代码写起来会遇到各种问题，到时候再见招拆招吧。


