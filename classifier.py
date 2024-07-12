import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import copy
from pandas.core.common import flatten


#Set Device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Show images
def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def visualize_augmentations(dataset, idx=0, samples=2, cols=2, random_img=False):
    dataset = copy.deepcopy(dataset)
    #we remove the normalize and tensor conversion from our augmentation pipeline
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
        
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            idx = np.random.randint(1,len(train_image_paths))
        image, lab = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(idx_to_class[lab])
    plt.tight_layout(pad=1)
    plt.show()   
    

#Create custom image dataset
class CandyDataset():
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        #print(f'in getitem, image file path is: {image_filepath} and idx is: {idx}')
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #cv2.imshow('image',image)
        #cv2.waitKey(0)
        

        label = image_filepath.split('\\')[-2]
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            
        return image, label

#Create our CNN
class Net(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) #This will halve the w, h, so 128x128
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*64*64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
    
            


#0 PREPARE THE DATA    
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5, 0.5))])

batch_size = 4

train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RandomCrop(height=256, width=256),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

'''
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True,
                                         transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True,
                                         transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')
'''

#Create Candy Dataset
train_data_path = 'C:\\Users\\ThomasWilk\\Pictures\\PyTorch\\train'
test_data_path = 'C:\\Users\\ThomasWilk\\Pictures\\PyTorch\\test'
train_image_paths = [] #list to store image paths
classes2 = [] #list to store classes

#Get all paths from train data path
for data_path in glob.glob(train_data_path + '\\*'):
    classes2.append(data_path.split('\\')[-1])
    train_image_paths.append(glob.glob(data_path + '/*'))
    
train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)
    
for x in range(len(train_image_paths)):
    pass
    #print(f'The train image paths are: {train_image_paths[x]}')
    
print(f'The number of classes is: {len(classes2)}')
print(f'First class is: {classes2[0]}')
print(f'Second class is: {classes2[1]}')
    
#Split train valid from train paths (80,20)
train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 

#Create test image paths
test_image_paths = []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))
    
test_image_paths = list(flatten(test_image_paths))

print(f'Train size is: {len(train_image_paths)}; Valid size is: {len(valid_image_paths)}; Test size is: {len(test_image_paths)}')

#Covert class names to indices

idx_to_class = {i:j for i, j in enumerate(classes2)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
print(class_to_idx)

#Create datasets
train_dataset = CandyDataset(train_image_paths,train_transforms)
valid_dataset = CandyDataset(valid_image_paths,test_transforms) #test transforms are applied
test_dataset = CandyDataset(test_image_paths,test_transforms)

#Print the size of a sample from the train dataset
#print(f'Size of train_dataset is: {len(train_dataset)}')

print(f'The shape of the 11th image in the train dataset is {train_dataset[10][0].shape}')
print(f'The label of the 11th image in the train dataset is {train_dataset[10][1]}')

#visualize_augmentations(train_dataset,np.random.randint(1,len(train_image_paths)), random_img = True)

#Set up DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

#Batch of image tensors
print(f'A batch of image tensors is: {next(iter(train_loader))[0].shape}')
print(f'A batch of image label tensors is: {next(iter(train_loader))[1].shape}')
      
'''
#Get random training images:
dataiter = iter(trainloader)
images, labels = next(dataiter)

#Show images
imshow(torchvision.utils.make_grid(images))

#Print labels
print(' '.join(f'{classes[labels[j]]:10s}' for j in range(batch_size)))
'''

#1 Create a Model (done above)
input_channels = 3 #3, for our color images
num_classes = 2 #currently just 2 classes

net = Net(input_channels=input_channels, num_classes=num_classes)

#2 Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#3 Train the network
epochs = 2
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

#4 Check accuracy and see how we did:
def check_accuracy(loader, model):
     
    num_correct = 0
    num_samples = 0
    net.eval()
    
    with torch.no_grad():
        for x, y in loader:
            #x = x.to(device=device)
            #y = y.to(device=device)
            
            scores = net(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    net.train()
print('Check accuracy with training images:')    
check_accuracy(train_loader, net)

print('Check accuracy with validation images:')    
check_accuracy(valid_loader, net)

print('Check accuracy with test images:')    
check_accuracy(test_loader, net)
    
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    

#Save model (change options later if needed)
save_path = 'C:\\Users\\ThomasWilk\\Pictures\\PyTorch\\model.pth'
torch.save(net.state_dict(), save_path)
