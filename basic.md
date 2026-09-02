## 学习目的

- 进行深度学习方面的实践，构建神经网络，超参数调整，正则化，诊断偏差，方差，高级优化算法

- 结构化机器学习工程

- 卷积神经网络，序列模型（RNN-LSTM），如何应用于自然语言处理（NLP）

# 1 二分类任务

二分类任务的目标是训练出来一个分类器

## 1.1参数

图片特征向量的维度：
$n_x=size×通道数$
输入：
$x∈R^{n_x}$
输出：
$y∈(0，1)$
样本的表示：
$(x^i,y^i)$
训练样本集合的表示：
$\{(x^1,y^1),(x^2,y^2),...,(x^i,y^i),...,(x^m,y^m)\}$
训练样本的个数(m)：
$m = m_{train}$
X的矩阵大小
$n_x×m$

<img width="996" height="710" alt="image" src="https://github.com/user-attachments/assets/410c8343-f70a-4f66-aef5-f9b8e191ddea" />

$X∈R^{n_x×m}$
Python命令：
$X.shape = (n_x,m)$
Y的矩阵大小为1×m

<img width="399" height="111" alt="image-20260901114122331" src="https://github.com/user-attachments/assets/ec5a8386-2189-43b5-88a4-2ee17fd162d1" />

Python命令：
$Y.shape = (1,m)$

## 1.2 logistic回归算法

logistic是一个用于二分分类的算法

预测值返回的一般不是布尔值，而是对正样本进行预测的可能性概率：
$\hat{y} = P(y=1|x),\hat{y}∈[0,1]$
Logistic回归的参数（parameters）是：
$w,w∈R^{n_x}\\b,b∈R$
Output:
$\hat{y}=w^Tx+b$

### 1.2.1 激活函数（sigmoid函数）

为了实现预测值介于[0,1]范围内，我们使用激活函数（sigmoid函数）作用于该函数，即
$\hat{y}=\sigma（w^Tx+b）$
<img width="375" height="195" alt="image-20260901115849021" src="https://github.com/user-attachments/assets/57ace69e-4bb7-4ad2-9931-4c52c89f4f25" />


sigmoid函数：
$\sigma(z) = \frac{1}{1 + e^{-z}}$
即：
$\hat{y}=\frac{1}{1 + e^{-{（w^Tx+b）}}}$

### 1.2.2损失函数&成本函数

训练达到好的效果意味着损失函数和成本函数的值尽可能小

- 损失函数衡量的是单个训练样本上的表现

- 成本函数衡量的是全体训练样本上的表现

训练集的目的，是为了达到通过训练，调整参数w和b,使得预测结果接近实际结果（0/1）
$\hat{y}^{(i)}≈{y}^{(i)}$
损失函数（误差函数）[Loss(error) function]，起着与误差平方相似作用
$L(\hat{y},y)=\frac{1}{2}(\hat{y}-y)^2$
因为使用误差平方会使得梯度下降可能只能找到局部最优但找不到全局最优解

<img width="285" height="178" alt="image-20260901122307132" src="https://github.com/user-attachments/assets/f4ec32c8-e49a-433f-8948-d5adae526256" />

所以选择一种损失函数：
$L(\hat{y},y)=-（ylog\hat{y}+(1-y)log(1-\hat{y})）$

<img width="991" height="154" alt="image-20260901122909608" src="https://github.com/user-attachments/assets/3c47840c-433c-4cd5-9e9c-35cda391474b" />


成本函数（Cost function）：
${ J ( w , b ) } =\frac { 1 } { m } \sum _ { i = 1 } ^ { m } L ( \hat { y } ^ { ( i ) } , y ^ { ( i ) } ) = - \frac { 1 } { m } \sum _ { i = 1 } ^ { m } [ y ^ { ( i ) } log \hat { y } ^ { ( i ) } + ( 1 - y ^ { ( i ) } ) log ( 1 - \hat { y } ^ { ( i ) } ) ]$
