import dataloaders
from model.modules import ConvLista_T, ListaParams
import torch
import numpy as np
from tqdm import tqdm
import argparse
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--stride", type=int, dest="stride", help="stride size", default=8)
parser.add_argument("--num_filters", type=int, dest="num_filters", help="Number of filters", default=175)
parser.add_argument("--kernel_size", type=int, dest="kernel_size", help="The size of the kernel", default=11)
parser.add_argument("--threshold", type=float, dest="threshold", help="Init threshold value", default=0.01)
parser.add_argument("--noise_level", type=int, dest="noise_level", help="Should be an int in the range [0,255]", default=25)
parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=2e-4)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="Learning rate decrease step", default=50)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.35)
parser.add_argument("--eps", type=float, dest="eps", help="ADAM epsilon parameter", default=1e-3)
parser.add_argument("--unfoldings", type=int, dest="unfoldings", help="Number of LISTA unfoldings", default=12)
parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Total number of epochs to train", default=250)
parser.add_argument("--patch_size", type=int, dest="patch_size", help="Total number of epochs to train", default=128)
parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results' dir path", default='trained_models')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.", default=None)
parser.add_argument("--data_path", type=str, dest="data_path", help="Path to the dir containing the training and testing datasets.", default="./datasets/")
args = parser.parse_args()


test_path = [f'{args.data_path}/BSD68/']
train_path = [f'{args.data_path}/CBSD432/',f'{args.data_path}/waterloo/']
kernel_size = args.kernel_size
stride = args.stride
num_filters = args.num_filters
lr = args.lr
eps = args.eps
unfoldings = args.unfoldings
lr_decay = args.lr_decay
lr_step = args.lr_step
patch_size = args.patch_size
num_epochs = args.num_epochs
noise_std = args.noise_level / 255
threshold = args.threshold

params = ListaParams(kernel_size, num_filters, stride, unfoldings)
loaders = dataloaders.get_dataloaders(train_path, test_path, patch_size, 1)
model = ConvLista_T(params).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)

psnr = {x: np.zeros(num_epochs) for x in ['train', 'test']}

guid = args.model_name if args.model_name is not None else uuid.uuid4()

config_dict = {
    'uuid': guid,
    'kernel_size':kernel_size,
    'stride': stride,
    'num_filters': num_filters,
    'lr':lr,
    'unfoldings': unfoldings,
    'lr_decay': lr_decay,
    'patch_size': patch_size,
    'num_epochs': num_epochs,
    'lr_step': lr_step,
    'eps': eps,
    'threshold': threshold,
    'noise_std': noise_std,
         }
print(config_dict)
with open(f'{args.out_dir}/{guid}.config','w') as txt_file:
    txt_file.write(str(config_dict))


print('Training model...')
for epoch in tqdm(range(num_epochs)):
    for phase in ['train', 'test']:
        if phase == 'train':
            scheduler.step()
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        # Iterate over data.
        num_iters = 0
        for batch in loaders[phase]:
            batch = batch.cuda()
            noise = torch.randn_like(batch) * noise_std
            noisy_batch = batch + noise

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                output = model(noisy_batch)
                loss = (output - batch).pow(2).sum() / batch.shape[0]

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            psnr[phase][epoch] += -10*np.log10(loss.item() / (batch.shape[2]*batch.shape[3]))
            num_iters += 1
        psnr[phase][epoch] /= num_iters
        print(f'{phase} PSNR: {psnr[phase][epoch]}')
        with open(f'{args.out_dir}/{guid}_{phase}.psnr','a') as psnr_file:
            psnr_file.write(f'{psnr[phase][epoch]},')
    # deep copy the model
    torch.save(model.state_dict(), f'{args.out_dir}/{guid}.model')
