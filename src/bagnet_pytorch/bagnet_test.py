import pytorchnet as pytorchnet
import torch
import torch.nn as nn
import torchvision

model = pytorchnet.bagnet17(pretrained=True)

# ===========================
# optimizer and loss funtion:
# ===========================
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

# You'll use the PyTorch torchvision class to load the data.

# The Torchvision library includes several popular datasets
# such as Imagenet, CIFAR10, MNIST, etc,
# model architectures, and common image transformations for computer vision.
# That makes data loading in Pytorch quite an easy process.


# Loading and normalizing the data.
# Define transformations for the training and test sets
transformations = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# CIFAR10 dataset consists of 50K training images.
# We define the batch size of 10 to load 5,000 batches of images.
batch_size = 10
number_of_labels = 10

# Create an instance for training.
# When we run this code for the first time,
# the CIFAR10 train dataset will be downloaded locally.
train_set = CIFAR10(
    root="./data", train=True, transform=transformations, download=True
)

# Create a loader for the training set
# which will read the data within batch size and put into memory.
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=0
)
print(
    "The number of images in a training set is: ",
    len(train_loader) * batch_size,
)

# Create an instance for testing, note that train is set to False.
# When we run this code for the first time,
# the CIFAR10 test dataset will be downloaded locally.
test_set = CIFAR10(
    root="./data", train=False, transform=transformations, download=True
)

# Create a loader for the test set
# which will read the data within batch size and put into memory.
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=0
)
print("The number of images in a test set is: ", len(test_loader) * batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# Define the loss function with Classification Cross-Entropy loss
# and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

from torch.autograd import Variable


# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)


# ===============================
# TEST FUNCTION:
# ===============================
# Function to test the model with the test dataset
# and print the accuracy for the test images
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = 100 * accuracy / total
    return accuracy


# ============================
# TRAIN FUNCTION:
# ============================
# Training function. We simply have to loop over our data iterator
# and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs
            images = Variable(images.to(device))
            # images = images.to(device)
            labels = Variable(labels.to(device))
            # labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()  # extract the loss value
            if i % 1000 == 999:
                # print every 1000 (twice per epoch)
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / 1000)
                )
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch
        # when tested over all 10000 test images
        accuracy = testAccuracy()
        print(
            "For epoch",
            epoch + 1,
            "the test accuracy over the whole test set is %d %%" % (accuracy),
        )

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


import matplotlib.pyplot as plt
import numpy as np


# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))

    # Show the real labels on the screen
    print(
        "Real labels: ",
        " ".join("%5s" % classes[labels[j]] for j in range(batch_size)),
    )

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels.
    # The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print(
        "Predicted: ",
        " ".join("%5s" % classes[predicted[j]] for j in range(batch_size)),
    )


if __name__ == "__main__":
    # Let's build our model
    train(200)
    print("Finished Training")

    # Test which classes performed well
    testAccuracy()

    # Let's load the model we just created and test the accuracy per label
    model = pytorchnet.bagnet17(pretrained=True)
    path = "myFirstModel.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch()
