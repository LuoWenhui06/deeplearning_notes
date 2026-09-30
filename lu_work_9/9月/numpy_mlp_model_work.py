import numpy as np
from sklearn.metrics import accuracy_score, f1_score

np.random.seed(42)
batch_size = 64
epochs = 200
lr = 0.1

def image(image_path, num):
    with open(image_path, "rb") as f:
        f.seek(16)
        image = np.frombuffer(f.read(), dtype=np.uint8)
        image = image.reshape(num, 784)
        x = image / 255.0
    return x

def label(label_path):
    with open(label_path, "rb") as f:
        f.seek(8)
        label = np.frombuffer(f.read(), dtype=np.uint8)
        y = np.eye(10)[label]
    return y

x_train = image(r"C:\Users\16131\Desktop\陈学姐作业\1_MLP\data\MNIST\raw\train-images-idx3-ubyte",60000)
y_train = label(r"C:\Users\16131\Desktop\陈学姐作业\1_MLP\data\MNIST\raw\train-labels-idx1-ubyte")

class MLP:
    def __init__(self, layer_num, input_dims, output_dims):
        self.w = []
        self.b = []
        for i in range(layer_num):
            w = np.random.randn(input_dims[i], output_dims[i])
            b = np.zeros(output_dims[i])
            self.w.append(w)
            self.b.append(b)

    def forward(self, x):
        self.a = []
        self.z = []
        a = x
        self.a.append(a)
        for i in range(len(self.w)):
            z = a @ self.w[i] + self.b[i]
            self.z.append(z)
            if i != len(self.w) - 1:
                a = 1 / (1 + np.exp(-z))
            else:
                exp_a = np.exp(z - np.max(z, axis=1, keepdims=True))#减最大值，防溢出
                a = exp_a / np.sum(exp_a, axis=1, keepdims=True)
            self.a.append(a)
        return a
    
    def backward(self, dz, lr):
        m = dz.shape[0]
        for i in reversed(range(len(self.w))):
            dw = self.a[i].T @ dz / m
            db = np.sum(dz, axis=0) / m
            if i > 0:
                da = dz @ self.w[i].T
                dz =  da * self.a[i] * (1 - self.a[i])
            self.w[i] -= lr * dw
            self.b[i] -= lr * db

layer_num = 4
input_dims = [784, 255, 255, 255]
output_dims = [255, 255, 255, 10]
model = MLP(layer_num, input_dims, output_dims)

x_test =image(r"C:\Users\16131\Desktop\陈学姐作业\1_MLP\data\MNIST\raw\t10k-images-idx3-ubyte",10000)
y_test = label(r"C:\Users\16131\Desktop\陈学姐作业\1_MLP\data\MNIST\raw\t10k-labels-idx1-ubyte")

def save(model, x, y):
    y_hat = model.forward(x)
    pred = np.argmax(y_hat, axis=1)
    true = np.argmax(y, axis=1)
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="macro")
    return acc, f1

loss_history = []
acc_history = []
f1_history = []

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        y_hat = model.forward(x_batch)
        loss = np.mean((y_hat - y_batch) ** 2)
        epoch_loss += loss
        dy_hat = 2 * (y_hat - y_batch) / y_hat.shape[1]
        dz = y_hat * (dy_hat - np.sum(dy_hat * y_hat, axis=1, keepdims=True))
        model.backward(dz, lr)

    batch_num = int(np.ceil(len(x_train) / batch_size))
    epoch_loss /= batch_num
    loss_history.append(epoch_loss)
    acc, f1 = save(model, x_test, y_test)
    acc_history.append(acc)
    f1_history.append(f1)
    print(f"Epoch: {epoch + 1}/{epochs} Loss: {epoch_loss:.6f} Acc: {acc:.4f} F1: {f1:.4f}")


