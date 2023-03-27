import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Inspect the images in the dataset 
# import cv2
# import numpy as np

class DigitRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        # layers
        '''
        class torch.nn.Conv2d():
			Applies a 2D convolution over an input signal composed of several input planes.
        
			Parameters: 
				in_channels (int) : Number of channels in the input image;
    			out_channels (int) : Number of channels produced by the convolution;
				kernel_size (int or tuple) : Size of the convolving kernel;
        
        class torch.nn.Conv2d():
			Applies a linear transformation to the incoming data.
        
			Parameters: 
				in_features (int) : size of each input sample;
    			out_features (int) : size of each output sample;
        '''
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1   = nn.Linear(20 * 10 * 10, 500)
        self.fc2   = nn.Linear(500, 10)
        
    def forward(self, x):
        batch_size = x.size(0)
        # input : batch size * 1 * 28 * 28
        x = self.conv1(x)
        # (length - kernel size + 2 * padding) / stride + 1 24 - 13
        # output: batch size * 10 * 24 * 24
        x = F.relu(x)
        # input : batch size * 10 * 24 * 24
        x = F.max_pool2d(x, 2, 2)
        # output: batch size * 10 * 12 * 12
        
        # input : batch size * 10 * 12 * 12
        x = self.conv2(x)
        # output: batch size * 20 * 10 * 10
        x = F.relu(x)
        
        # dim = 20 * 10 * 10 = 2000
        x = x.view(batch_size, -1)
        
        # input : batch size * 2000
        x = self.fc1(x)
        # output: batch size * 500
        x = F.relu(x)
        # input : batch size * 500
        x = self.fc2(x)
        # output: batch size * 10
        
        output = F.log_softmax(x, dim=1)
        return output
    

def train_model(model, device, train_loader, optimizer, epoch):
    # Sets the module in training mode.
    model.train()
    # batch_size = 100, total 5000 batch.
    # image.shape = [100, 1, 28, 28], label.shape = [100]
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        # Set the gradient of the parameters in the model to 0.
        optimizer.zero_grad()
        # Forward propagation for predicted value.
        output = model(image)
        # Calculate loss.
        # The target input of CrossEntropyLoss() isn't one-hot encoding format but category value.
        loss = F.cross_entropy(output, label)
        # pred = output.argmax(dim=1)
        # Backward propagation for gradient, calculate the gradient value of each parameter.
        loss.backward()
        # Update the values of parameters by gradient descent.
        optimizer.step()
        if batch_idx % 2500 == 0:
            print("Train Epoch: {} \t Loss: {:0.6f}".format(epoch, loss.item()))
          
            
def evalu_model(model, device, test_loader):
    # Sets the module in evaluating mode.
    model.eval()
    accuracy  = .0
    test_loss = .0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            test_loss += F.cross_entropy(output, label).item()
            # pred.shap = [100]
            pred = output.argmax(dim=1)
            accuracy += pred.eq(label).sum().item()
            # accuracy += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test average loss: {:.4f} \t Accuracy: {:0.3f}".format(test_loss, 100.0 * accuracy / len(test_loader.dataset)))


if __name__ == "__main__":

    BATCH_SIZE = 100
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    torchvision.transforms
        Perform some manipulation of the data and make it suitable for training.
  
    class torchvision.transforms.Compose(transforms):
        Composes several transforms together.

        Parameters: 
        	transforms : List of transforms to compose.
        
    class transforms.ToTensor():
        Converts a PIL image or NumPy ndarray into a FloatTensor,
        and scales the image's pixel intensity values in the range [0, 1].
        
    class transforms.Normalize(mean, std):
        Normalize the images channel by channel until mean = 0 and std = 1 to accelerate the convergence of the model.
        
        Parameters: 
            mean (sequence) : Sequence of means for each channel.
			std (sequence) : Sequence of standard deviations for each channel.
        
        why 0.1307 & 0.3081: 
            Images in MNIST has only one channel, so there are only one parameter in mean and std,
            and mean = 0.1307, std = 0.3081
    '''
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    '''
    torchvision.datasets
        Torchvision provides many built-in datasets in the torchvision.datasets module, such as MINST.
        
    class torchvision.datasets.MNIST():
		MNIST dataset [http://yann.lecun.com/exdb/mnist/].
        Parameters:
          	root (string) : Root directory of dataset;
            train (bool, optional) : If True, creates train dataset, else create test dataset;
         	download (bool, optional) : If True, downloads the dataset from the internet and puts it in root directory;
          	transform (callable, optional) : A function/transform that takes in an PIL image and returns a transformed version
        
    class torch.utils.data.DataLoader():
        Combines a dataset and a sampler, and provides an iterable over the given dataset.
        Parameters:
            dataset (Dataset) : Dataset from which to load the data;
            batch_size (int, optional) : How many samples per batch to load;
            shuffle (bool, optional) : Set to True to have the data reshuffled at every epoch;
    '''
    train_data = datasets.MNIST("data",
                                train=True,
                                download=True,
                                transform=pipeline
                                )
    train_loader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True
                              )
    evalu_data = datasets.MNIST("data",
                               train=False,
                               download=True,
                               transform=pipeline
                               )
    evalu_loader  = DataLoader(evalu_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True
                               )

    # Inspect the images in the dataset 
    # 
    # with open("./data/MNIST/raw/train-images-idx3-ubyte", "rb") as f:
    #    file = f.read()
    # image = [int(str(item).encode('ascii'), 16) for item in file[16 : 16 + 784]]
    # image_np = np.array(image, dtype=np.uint8).reshape(28, 28, 1)
    # cv2.imwrite("digit.jpg", image_np)
    
    '''
    DigitRecognition() : 
        Designed neural networks.
    .to() :
        Copy all the tensor variables to the GPU (or CPU), and all subsequent operations are performed on the assigned device.
    '''
    model = DigitRecognition().to(DEVICE)
    
    '''
    class torch.optim.Adam:
        Implements Adam algorithm.
        Parameters:
            params (iterable) :  iterable of parameters to optimize or dicts defining parameter groups.
            
    model.parameters() returns all the parameters of the model and passes them into the Adam function to construct an Adam optimizer.
    '''
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range (1, EPOCHS + 1):
        train_model(model, DEVICE, train_loader, optimizer, epoch)
        evalu_model(model, DEVICE, evalu_loader)
    
