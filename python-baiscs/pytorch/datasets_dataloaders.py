# This is the python file to practice with dataloaders and datasets for numpy

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
        root="~/ml/generating-model/python-baiscs/pytorch",    # the path where the train/test data is stored
        train=True,     # specifies training or test dataset
        download=True,  # downloads the data from the internet if it’s not available at root
        transform=ToTensor()    # transform and target_transform specify the feature and label transformations
)

test_data = datasets.FashionMNIST(
        root="~/ml/generating-model/python-baiscs/pytorch",
        train=True,
        download=True,
        transform=ToTensor()
)

test_data = datasets.FashionMNIST(
        root="~/ml/generating-model/python-baiscs/pytorch",
        train=False,
        download=True,
        transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
plt.savefig("output.png")

