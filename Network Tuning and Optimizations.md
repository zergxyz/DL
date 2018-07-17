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
