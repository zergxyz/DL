### 训练、验证、测试集 (Train / Dev / Test sets)
对于一个需要解决的问题的样本数据，在建立模型的过程中，我们会将问题的data划分为以下几个部分：
* 训练集（train set）：用训练集对算法或模型进行训练过程；
* 验证集（development set）：利用验证集或者又称为简单交叉验证集（hold-out cross validation set）进行交叉验证，选择出最好的模型；
* 测试集（test set）：最后利用测试集对模型进行测试，获取模型运行的无偏估计。

在小数据量的时代，如：100、1000、10000的数据量大小，可以将data做以下划分：
* 无验证集的情况：70% / 30%；
* 有验证集的情况：60% / 20% / 20%；

大数据时代：验证集的目的是为了验证不同的算法哪种更加有效，所以验证集只要足够大能够验证大约2-10种算法哪种更好就足够了，不需要使用20%的数据作为验证集。如百万数据中抽取1万的数据作为验证集就可以了。测试集的主要目的是评估模型的效果，如在单个分类器中，往往在百万级别的数据中，我们选择其中1000条数据足以评估单个模型的效果。
* 100万数据量：98% / 1% / 1%；
* 超百万数据量：99.5% / 0.25% / 0.25%（或者99.5% / 0.4% / 0.1%）

### Bias / Variance 
High variance (overfitting) for example:
* Training error: 1%
* Dev error: 11%

high Bias (underfitting) for example:
* Training error: 15%
* Dev error: 14% 

high Bias (underfitting) && High variance (overfitting) for example:
* Training error: 15%
* Test error: 30%

Best:
* Training error: 0.5%
* Test error: 1%
These Assumptions came from that human has 0% error. If the problem isn't like that you'll need to use human error as baseline.




If your algorithm has a high bias:Try to make your NN bigger (size of hidden units, number of layers). Try a different model that is suitable for your data. Try to run it longer. Different (advanced) optimization algorithms.

If your algorithm has a high variance:
More data. Try regularization. Try a different model that is suitable for your data. You should try the previous two points until you have a low bias and low variance. In the older days before deep learning, there was a "Bias/variance tradeoff". But because now you have more options/tools for solving the bias and variance problem its really helpful to use deep learning. 

Training a bigger neural network never hurts. 

### Regularization
利用正则化来解决High variance 的问题，正则化是在 Cost function 中加入一项正则化项，惩罚模型的复杂度。

Logistic regression 加入正则化项的cost：

$J(\theta_0, \theta_1) = \dfrac {1}{2m} \displaystyle \sum {i=1}^m \left ( \hat{y}{i}- y_{i} \right)^2 = \dfrac {1}{2m} \displaystyle \sum {i=1}^m \left (h\theta (x_{i}) - y_{i} \right)^2$
L2 norm：
$$\dfrac{\lambda}{2m}||w||^{2} = \dfrac{\lambda}{2m}\sum\limits_{j=1}^{n_{x}} w_{j}^{2}=\dfrac{\lambda}{2m}w^{T}w$$
$$\dfrac {\lambda}{2m} ||w||_{2}^{2} = \dfrac {\lambda}{2m} \sum \limits_{j=1}^{n_{x}} w_{j}^{2}=\dfrac{\lambda}{2m}w^{T}w$$

L1正则化： $\dfrac{\lambda}{2m}||w||_{1}=\dfrac{\lambda}{2m}\sum\limits_{j=1}^{n_{x}}|w_{j}|$

Newral network cost: 
$$J(w^{[1]},b^{[1]},\cdots,w^{[L]},b^{[L]})=\dfrac{1}{m}\sum\limits_{i=1}^{m}l(\hat y^{(i)},y^{(i)})+\dfrac{\lambda}{2m}\sum\limits_{l=1}^{L}||w^{[l]}||_{F}^{2}$$ 
其中 $$||w^{[l]}||_{F}^{2}=\sum\limits_{i=1}^{n^{[l-1]}}\sum\limits_{j=1}^{n^{[l]}}(w_{ij}^{[l]})^{2}$$ ，因为 w 的大小为 (n^{[l-1]},n^{[l]}) ，该矩阵范数被称为“Frobenius norm”

Weight decay

在加入正则化项后，梯度变为：
$$dW^{[l]} = (form\_backprop)+\dfrac{\lambda}{m}W^{[l]}$$

则梯度更新公式变为：
$$W^{[l]}:= W^{[l]}-\alpha dW^{[l]}$$

代入可得：
$$W^{[l]}:= W^{[l]}-\alpha [ (form\_backprop)+\dfrac{\lambda}{m}W^{[l]}]\\ = W^{[l]}-\alpha\dfrac{\lambda}{m}W^{[l]} -\alpha(form\_backprop)\\=(1-\dfrac{\alpha\lambda}{m})W^{[l]}-\alpha(form\_backprop)$$

其中， $(1-\dfrac{\alpha\lambda}{m})$ 为一个 <1 的项，会给原来的 W^{[l]}一个衰减的参数，所以L2范数正则化也被称为权重衰减(Weight decay)。

Dropout 正则化
Dropout（随机失活）就是在神经网络的Dropout层，为每个神经元结点设置一个随机消除的概率，对于保留下来的神经元，我们得到一个节点较少，规模较小的网络进行训练。

实现Dropout的方法：反向随机失活（Inverted dropout）

首先假设对 layer 3 进行dropout：
``` python 
keep_prob = 0.8  # 设置神经元保留概率
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3)
a3 /= keep_prob
/= keep_prob
```
依照例子中的 keep_prob = 0.8 ，那么就有大约20%的神经元被删除了，也就是说 $a^{[3]}$ 中有20%的元素被归零了，在下一层的计算中有 $Z^{[4]}=W^{[4]}\cdot a^{[3]}+b^{[4]}$ ，所以为了不影响 $Z^{[4]}$ 的期望值，所以需要 $W^{[4]}\cdot a^{[3]}$ 的部分除以一个keep_prob。

Inverted dropout 通过对“a3 /= keep_prob”,则保证无论 keep_prob 设置为多少，都不会对 Z^{[4]} 的期望值产生影响。

Notation：在测试阶段不要用dropout，因为那样会使得预测结果变得随机。




