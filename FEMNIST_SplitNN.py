import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn, optim

import matplotlib.pyplot as plt
import numpy as np

import datas
import TrainTest
import classes


device = 'mps'
transform = transforms.Normalize((-0.5), (0.5))



client_model, shadow_model = classes.SplitNN().to(device), classes.ShadowNN().to(device)


client_train_ds = datas.FEMNIST(train=True, transform = transform, client_num = 1)
client_train_nums = datas.only_numbers(client_train_ds)
client_test_ds = datas.FEMNIST(train=False, transform = transform, client_num = 1)
client_test_nums = datas.only_numbers(client_test_ds)

shadow_train_ds = datas.FEMNIST(train=True, transform = transform, client_num = 2)
shadow_train_nums = datas.only_numbers(shadow_train_ds)
shadow_test_ds = datas.FEMNIST(train=False, transform = transform, client_num = 2)
shadow_test_nums = datas.only_numbers(shadow_test_ds)



# # Convert the tensor image to a numpy array
# image_np = image.squeeze().numpy()

# # Convert the label tensor to a Python scalar
# label = label.item()

# # Display the image and label
# plt.imshow(image_np, cmap='gray')
# plt.title(f"Label: {label}")
# plt.show()


client_loaders = {
    'train' : DataLoader(client_train_nums, batch_size=64, shuffle=True),
    'test'  : DataLoader(client_test_nums, batch_size=64,  shuffle=True),
}
shadow_loaders = {
    'train' : DataLoader(shadow_train_nums, batch_size=64, shuffle=True),
    'test'  : DataLoader(shadow_test_nums, batch_size=64,  shuffle=True),
}

client_criterion = nn.NLLLoss()
client_optimizer = optim.SGD(client_model.parameters(), lr=0.003, momentum=0.9)

shadow_criterion = nn.NLLLoss()
shadow_optimizer = optim.SGD(client_model.parameters(), lr=0.003, momentum=0.9)
epochs = 12

TrainTest.train(epochs, client_model, client_loaders['train'], client_criterion, client_optimizer)
TrainTest.test(client_model, client_loaders['test'])
TrainTest.train(epochs, shadow_model, shadow_loaders['train'], shadow_criterion, shadow_optimizer)
TrainTest.test(shadow_model, shadow_loaders['test'])

# save models
torch.save(client_model.state_dict(), "Trained_models/FEMNIST_SplitNN_client_nums.pth")
print("Saved PyTorch Model State to FEMNIST_SplitNN_client.pth")

torch.save(client_model.state_dict(), "Trained_models/FEMNIST_SplitNN_shadow_nums.pth")
print("Saved PyTorch Model State to FEMNIST_SplitNN_shadow.pth")

