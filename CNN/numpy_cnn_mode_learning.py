import numpy as np
import time

#一、获取数据
import os
def local_mnist(data_dir=r"C:\Users\16131\Desktop\陈学姐作业\1_MLP\data\MNIST\raw"):
    #定义文件名映射
    files = {
        'train_images': 'train-images-idx3-ubyte',
        'train_labels': 'train-labels-idx1-ubyte',  
        'test_images':  't10k-images-idx3-ubyte',
        'test_labels':  't10k-labels-idx1-ubyte'    
    }

    #检查文件是否存在
    for name, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"没找到文件: {path}")
        
#本地无数据就下载数据
#from sklearn.datasets import fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
#from sklearn.model_selection import train_test_split

    #定义图片解析函数，二进制转三维数组（通道，高，宽）
    def parse_images(filepath):
        with open(filepath, 'rb') as f:
            magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>u4')
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return images.astype(np.float32) / 255.0

    #定义标签解析函数，二进制转整数索引
    def parse_labels(filepath):
        with open(filepath, 'rb') as f:
            magic, num = np.frombuffer(f.read(8), dtype='>u4')
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
    #定义one_hot函数
    def one_hot(y, num_classes=10):
        return np.eye(num_classes)[y]

    #解析二进制数据
    X_train = parse_images(os.path.join(data_dir, files['train_images']))
    y_train = parse_labels(os.path.join(data_dir, files['train_labels']))
    X_test  = parse_images(os.path.join(data_dir, files['test_images']))
    y_test  = parse_labels(os.path.join(data_dir, files['test_labels']))

    #图片三维数组(B, H, W)转四维(B, C, H, W)（每批次样本数，通道数，高，宽）
    X_train = X_train.reshape(-1, 1, 28, 28)#-1，自己统计，不用人为指定
    X_test  = X_test.reshape(-1, 1, 28, 28)

    #标签 one-hot 编码
    y_train_oh = one_hot(y_train)
    y_test_oh  = one_hot(y_test)

    #打印验证
    print(f"训练集: {X_train.shape}, 标签: {y_train_oh.shape}")
    print(f"测试集: {X_test.shape},  标签: {y_test_oh.shape}")

    return (X_train, y_train_oh), (X_test, y_test_oh), y_train, y_test

#二、卷积层
##卷积实现（前向+反向）
###前向
def im2col(images,kernel_h,kernel_w,stride,padding):
    N, C, H, W = images.shape

    #零填充
    if padding > 0:
        images = np.pad(images, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')

    #计算卷积后的out形状
    _, _, H_pad, W_pad = images.shape
    out_h = (H_pad - kernel_h) // stride + 1
    out_w = (W_pad - kernel_w) // stride + 1

    # 高效实现卷积，把"滑动窗口逐个计算"转化为"一次矩阵乘法"。
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))
    for i in range(kernel_h):
        i_max = i + stride * out_h
        for j in range(kernel_w):
            j_max = j + stride * out_w
            col[:, :, i, j, :, :] = images[:, :, i:i_max:stride, j:j_max:stride]  

    #col重排序
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    #六维数组转二维矩阵
    return col, out_h, out_w  

###反向
def col2im(cols, images_shape, kernel_h, kernel_w, stride=1, padding=0):
    N, C, H, W = images_shape
    if padding > 0:
        H_pad = H + 2 * padding
        W_pad = W + 2 * padding
    else:
        H_pad, W_pad = H, W

    out_h = (H_pad - kernel_h) // stride + 1
    out_w = (W_pad - kernel_w) // stride + 1

    cols_reshaped = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w)
    cols_reshaped = cols_reshaped.transpose(0, 3, 4, 5, 1, 2)  # (N, C, kh, kw, out_h, out_w)

    images_padded = np.zeros((N, C, H_pad, W_pad))
    for i in range(kernel_h):
        i_max = i + stride * out_h
        for j in range(kernel_w):
            j_max = j + stride * out_w
            images_padded[:, :, i:i_max:stride, j:j_max:stride] += cols_reshaped[:, :, i, j, :, :]

    if padding > 0:
        return images_padded[:, :, padding:-padding, padding:-padding]
    return images_padded

##卷积层参数更新（前向+反向）
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        #初始化
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))#计算标准差
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale#生成随机权重，再用标准差缩放，获得初始权重w
        self.b = np.zeros((1, out_channels, 1, 1))#初始化偏置b
        # 梯度
        self.dW = None
        self.db = None

#参数w,b的前向更新
    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # im2col
        self.col, _, _ = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.padding)
        w_flat = self.W.reshape(self.W.shape[0], -1) 
        out = self.col @ w_flat.T + self.b.reshape(1, -1)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out
    
#参数w,b的反向梯度计算
    def backward(self, dout):
        N, out_c, out_h, out_w = dout.shape
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, out_c)
        w_flat = self.W.reshape(out_c, -1)
        self.dW = (dout_reshaped.T @ self.col).reshape(self.W.shape)
        self.db = np.sum(dout_reshaped, axis=0).reshape(1, out_c, 1, 1)
        dcol = dout_reshaped @ w_flat
        # col2im 还原
        dx = col2im(dcol, self.x.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
        return dx
    
#三、激活层（ReLU激活函数）
##ReLU激活函数
def relu(z):
    return np.maximum(0, z)#计算过程即为该激活函数的表达式
##ReLU激活函数导数
def relu_derivative(z):
    return (z > 0).astype(np.float32)#计算过程即为该激活函数导数的表达式

#激活层
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx
    
#四、池化层
class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        col, _, _ = im2col(x, self.pool_size, self.pool_size, self.stride, 0)
        col = col.reshape(-1, C, self.pool_size * self.pool_size)
        self.max_idx = col.argmax(axis=2)  
        out = col.max(axis=2)  
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.out_shape = (N, C, out_h, out_w)
        self.n_patches = N * out_h * out_w
        return out

    def backward(self, dout):
        N, C, out_h, out_w = dout.shape
        n_patches = N * out_h * out_w
        pool_area = self.pool_size * self.pool_size
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(n_patches, C)
        dcol = np.zeros((n_patches, C, pool_area))
        patch_idx = np.arange(n_patches)
        for c in range(C):
            dcol[patch_idx, c, self.max_idx[:, c]] = dout_flat[:, c]
        dcol = dcol.reshape(n_patches, -1)
        dx = col2im(dcol, self.x.shape, self.pool_size, self.pool_size, self.stride, 0)
        return dx

#五、展平层
class Flatten:   
    def __init__(self):
        self.shape = None

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)

#六、全连接层
class Dense:
 
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)  # He 初始化
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features))
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W.T
        return dx

#七、Softmax激活函数
def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)#防溢出
    exp_z = np.exp(z_shifted)#计算以e为底的指数
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)#计算过程即为该激活函数的表达式

#八、多分类交叉熵损失函数
def cross_entropy_loss(probs, y_onehot):
    N = probs.shape[0]#获取当前批次样本数量
    probs_clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)#防止 log(0)
    loss = -np.sum(y_onehot * np.log(probs_clipped)) / N#计算过程即为该损失函数的表达式
    return loss

#九、CNN流程
class CNN:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.conv1 = Conv2D(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)
        self.conv2 = Conv2D(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)
        self.flatten = Flatten()
        self.fc1 = Dense(32 * 7 * 7, 128)
        self.relu3 = ReLU()
        self.fc2 = Dense(128, 10)

        self.layers = [self.conv1, self.relu1, self.pool1,
                       self.conv2, self.relu2, self.pool2,
                       self.flatten, self.fc1, self.relu3, self.fc2]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x  # logits

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update(self):
        for layer in self.layers:
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    def predict(self, x):
        logits = self.forward(x)
        return softmax(logits)

#十、训练
def train(model, X_train, y_train_oh, X_test, y_test,y_train_raw, 
          epochs=5, batch_size=64, lr=0.01, lr_decay=0.95):
    N = X_train.shape[0]
    model.lr = lr

    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_oh[indices]

        epoch_loss = 0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # 前向传播
            logits = model.forward(X_batch)
            probs = softmax(logits)

            # 计算损失
            loss = cross_entropy_loss(probs, y_batch)
            epoch_loss += loss
            n_batches += 1

            # 反向传播
            dout = (probs - y_batch) / X_batch.shape[0]
            model.backward(dout)

            # 更新参数
            model.update()

        # 学习率衰减
        model.lr *= lr_decay

        # 每个 epoch 评估
        avg_loss = epoch_loss / n_batches
        train_acc = evaluate(model, X_train[:5000], y_train_raw[:5000])
        test_acc  = evaluate(model, X_test, y_test)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc (5k): {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"LR: {model.lr:.6f}")

def evaluate(model, X, y):
    #评估准确率
    correct = 0
    total = X.shape[0]
    batch_size = 256
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        probs = model.predict(X[start:end])
        preds = np.argmax(probs, axis=1)
        correct += np.sum(preds == y[start:end])
    return correct / total

    #运行训练
if __name__ == "__main__":
    np.random.seed(42)

    print("="*60)
    print("正在构建 CNN 模型...")
    print("网络结构:")
    print("  Conv2D(1->16, 3x3, pad=1) -> ReLU -> MaxPool(2x2)")
    print("  Conv2D(16->32, 3x3, pad=1) -> ReLU -> MaxPool(2x2)")
    print("  Flatten -> Dense(1568->128) -> ReLU -> Dense(128->10)")
    print("="*60)

    #加载数据
    (X_train, y_train_oh), (X_test, y_test_oh), y_train_raw, y_test_raw = local_mnist()

    model = CNN(learning_rate=0.01)

    USE_SUBSET = True
    if USE_SUBSET:
        SUBSET_TRAIN = 10000  
        SUBSET_TEST  = 2000   
        X_tr = X_train[:SUBSET_TRAIN]
        y_tr_oh = y_train_oh[:SUBSET_TRAIN]
        y_tr_raw = y_train_raw[:SUBSET_TRAIN] 
        X_te = X_test[:SUBSET_TEST]
        y_te = y_test_raw[:SUBSET_TEST]
        y_te_oh = y_test_oh[:SUBSET_TEST]
        print(f"\n[子集模式] 训练: {SUBSET_TRAIN} 张, 测试: {SUBSET_TEST} 张")
    else:
        X_tr = X_train
        y_tr_oh = y_train_oh
        y_tr_raw = y_train_raw 
        X_te = X_test
        y_te = y_test_raw
        y_te_oh = y_test_oh

    print(f"\n开始训练...\n")
    t0 = time.time()
    train(model, X_tr, y_tr_oh, X_te, y_te,y_tr_raw,
          epochs=5, batch_size=64, lr=0.01, lr_decay=0.9)
    t1 = time.time()
    print(f"\n训练完成！总耗时: {t1-t0:.1f} 秒")

    # 最终测试
    final_acc = evaluate(model, X_te, y_te)
    print(f"最终测试准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")

    # 可视化预测结果
    print("\n--- 部分预测示例 ---")
    probs = model.predict(X_te[:20])
    preds = np.argmax(probs, axis=1)
    for i in range(20):
        mark = "✓" if preds[i] == y_te[i] else "✗"
        print(f"  样本 {i:2d}: 真实={y_te[i]}, 预测={preds[i]} {mark}")