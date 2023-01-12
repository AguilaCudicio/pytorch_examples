import torch
import torchvision
import torch.nn as nn

#based on https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/

print("Is cuda available?")
print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 10 classes (digits from 0 to 9)
batch_size = 32
num_classes = 10
learning_rate = 0.001
num_epochs = 10

train_dt = torchvision.datasets.MNIST(
    root = 'data',
    train = True,
    transform = torchvision.transforms.ToTensor(), 
    download = True     
)
test_dt = torchvision.datasets.MNIST(
    root = 'data',
    train = False,
    transform = torchvision.transforms.ToTensor(),
    download = True
)

#MNIST contains 60000 images for training, 10000 for testing.
#each image is a square with 28x28 pixels
loader_train = torch.utils.data.DataLoader(dataset = train_dt,
                                           batch_size = batch_size,
                                           shuffle = True)


loader_test = torch.utils.data.DataLoader(dataset = test_dt,
                                        batch_size = batch_size,
                                        shuffle = True)

class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        # 1 channels since they're grayscale
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=18, out_channels=34, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=34, out_channels=34, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(544, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

model = ConvNeuralNet(num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

total_step = len(loader_train)

for epoch in range(num_epochs):
	#Load in the data in batches using the loader_train object
    for i, (images, labels) in enumerate(loader_train):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in loader_train:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))

