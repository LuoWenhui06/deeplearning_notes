import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda")
epochs = 50
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

train_data = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long)), batch_size=64, shuffle=True)
test_data  = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long)), batch_size=256, shuffle=False)

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()  

        self.fc1 = torch.nn.Linear(784, 255)
        self.fc2 = torch.nn.Linear(255, 255)
        self.fc3 = torch.nn.Linear(255, 255)
        self.fc4 = torch.nn.Linear(255, 10)

    def forward(self, input):     
        out_0 = F.relu(self.fc1(input))   
        out_1 = F.relu(self.fc2(out_0))
        out_2 = F.relu(self.fc3(out_1))        
        out_3 = self.fc4(out_2)
        return out_3

def train():
    model.train()                             
    lossfunc = torch.nn.CrossEntropyLoss()    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
    for epoch in range(epochs):
        loss = 0.0
        correct = 0
        total = 0
        for image, label in train_data:
            image, label = image.to(device), label.to(device)  
            optimizer.zero_grad()
            output = model(image)
            loss_batch = lossfunc(output, label)
            loss_batch.backward()
            optimizer.step()
            loss += loss_batch.item() * image.size(0)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            right_num += (predicted == label).sum().item()
        avg_loss = loss / total
        accuracy = right_num / total
        print(f'Epoch: {epoch + 1} \tTrain Loss: {avg_loss:.6f} \tTrain Acc: {accuracy:.3f}')

def test():
    model.eval()
    right_num = 0
    total = 0
    test_loss = 0.0
    lossfunc = torch.nn.CrossEntropyLoss() 
    with torch.no_grad():
        for images, labels in test_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += lossfunc(outputs, labels).item() * images.size(0)
            _, No = torch.max(outputs.data, 1)
            total += labels.size(0)
            right_num += (No == labels).sum().item()
    avg_loss = test_loss / total
    accuracy = right_num / total
    print(f'Test Loss: {avg_loss:.6f} \tTest Acc: {accuracy:.3f}')
    return accuracy

model = MLP().to(device)                   

if __name__ == '__main__':
    train()
    test()