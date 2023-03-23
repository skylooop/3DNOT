from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm
from utils import *


from dataset import MNIST3D

dataset_2 = MNIST3D(number=2)
dataset_4 = MNIST3D(number=4)

batch_size = 64
train_dataloader_2 = DataLoader(dataset_2, batch_size=batch_size, shuffle=True)
train_dataloader_4 = DataLoader(dataset_4, batch_size=batch_size, shuffle=True)
sample_x_batch = next(iter(train_dataloader_2))
sample_y_batch = next(iter(train_dataloader_4))
print('Sample batch dimensions: ', sample_x_batch.shape)

from models import ResNet_D, UNet

T_net = UNet(3, 3) # strong case
f_net = ResNet_D(16, nc=3)

T_x = T_net(sample_x_batch)

f_y = f_net(sample_y_batch)
print('Y = T(X) shape: ', T_x.shape, 'Potential of Y shape: ', f_y.shape)

iter_3dmnist_2, iter_3dmnist_4 = iter(train_dataloader_2), iter(train_dataloader_4)

def sample_mnist_2():
    global iter_3dmnist_2, train_dataloader_2
    try:
        return next(iter_3dmnist_2)
    except StopIteration:
        iter_3dmnist_2 = iter(train_dataloader_2)
        return next(iter_3dmnist_2)

def sample_mnist_4():
    global iter_3dmnist_4, train_dataloader_4
    try:
        return next(iter_3dmnist_4)
    except StopIteration:
        iter_3dmnist_4 = iter(train_dataloader_4)
        return next(iter_3dmnist_4)

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def sq_cost (X, Y):
    return (X-Y).square().flatten(start_dim=1).mean(dim=1)

COST = sq_cost

print('Cost(X, Y) shape', COST(sample_x_batch, T_x).shape)

T_ITERS = 5
MAX_STEPS = 10000 + 1

T = UNet(3, 3) # strong case
f = ResNet_D(16, nc=3)

T.to(DEVICE)
f.to(DEVICE)

T_opt = torch.optim.Adam(T.parameters(), lr=1e-4, weight_decay=1e-10)
f_opt = torch.optim.Adam(f.parameters(), lr=1e-4, weight_decay=1e-10)


for step in tqdm(range(MAX_STEPS)):
    # T optimization
    T.train(True); f.eval()
    for t_iter in range(T_ITERS):
        X = sample_mnist_2()
        X = torch.tensor(X, device=DEVICE)
        T_loss = COST(X, T(X)).mean() - f(T(X)).mean()
        T_opt.zero_grad(); T_loss.backward(); T_opt.step()

    # f optimization
    T.eval(); f.train(True)
    X, Y = sample_mnist_2(), sample_mnist_4()
    X = torch.tensor(X, device=DEVICE)
    Y = torch.tensor(Y, device=DEVICE)
    f_loss = f(T(X)).mean() - f(Y).mean()
    f_opt.zero_grad(); f_loss.backward(); f_opt.step()
    print('T_loss: ', T_loss, 'f_loss: ', f_loss)
    if step % 200 == 0:
#         clear_output(wait=True)
        print("Step", step)

        # TODO: The code for plotting the results
        
