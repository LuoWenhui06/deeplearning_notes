import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda")
epochs = 100
batch_size = 64


def image(image_path, num):
    with open(image_path, "rb") as f:
        f.seek(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, 784).astype(np.float32) / 255.0
    return data

def label(label_path):
    with open(label_path, "rb") as f:
        f.seek(8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

x_train = image(r"C:\Users\16131\Desktop\deeplearning_learning\chen_work\1_MLP\data\MNIST\train-images-idx3-ubyte", 60000)
y_train = label(r"C:\Users\16131\Desktop\deeplearning_learning\chen_work\1_MLP\data\MNIST\train-labels-idx1-ubyte")
x_test  = image(r"C:\Users\16131\Desktop\deeplearning_learning\chen_work\1_MLP\data\MNIST\t10k-images-idx3-ubyte", 10000)
y_test  = label(r"C:\Users\16131\Desktop\deeplearning_learning\chen_work\1_MLP\data\MNIST\t10k-labels-idx1-ubyte")

x_train = x_train.reshape(-1, 1, 28, 28)
x_test  = x_test.reshape(-1, 1, 28, 28)

train_data = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long)),batch_size=64, shuffle=True)
test_data = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long)),batch_size=256, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, input):
        out_1 = self.pool(self.relu(self.conv1(input)))  # [N,32,14,14]
        out_2 = self.pool(self.relu(self.conv2(out_1)))  # [N,32,7,7]
        out_3 = self.pool(self.relu(self.conv3(out_2)))  # [N,64,3,3]
        out_3 = out_3.view(out_3.size(0), -1)                # [N,576]
        out_3 = self.relu(self.fc1(out_3))
        out_4 = self.fc2(out_3)
        return out_4

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in train_data:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in test_data:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total * 100
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_data):.4f} | Test Acc: {acc:.2f}%")