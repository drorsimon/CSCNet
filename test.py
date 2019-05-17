import torch
from model import utils, modules
import dataloaders
from os.path import join
from uuid import UUID
from matplotlib import pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, dest="root_folder", help="The trained model's dir path", default='./trained_models')
parser.add_argument("--model_name", type=str, dest="uuid", help="The model's name", required=True)
parser.add_argument("--data_path", type=str, dest="data_path", help="Path to the dir containing the training and testing datasets.", default="./datasets/")
args = parser.parse_args()

config_filename = args.uuid+'.config'
model_filename = args.uuid+'.model'
config_path = join(args.root_folder, config_filename)

with open(config_path) as conf_file:
    conf = conf_file.read()
conf = eval(conf)
params = modules.ListaParams(conf['kernel_size'], conf['num_filters'], conf['stride'], conf['unfoldings'])
model = modules.ConvLista_T(params)
model.load_state_dict(torch.load(join(args.root_folder, model_filename)))  # cpu is good enough for testing

test_path = [f'{args.data_path}/Set12/']
# test_path = ['../../../../images/BSD68/']
loaders = dataloaders.get_dataloaders([], test_path, 128, 1)
loaders['test'].dataset.verbose = True
model.eval()   # Set model to evaluate mode
model.cuda()
num_iters = 0
noise_std = conf['noise_std']
psnr = 0
print(f"Testing model: {args.uuid} with noise_std {noise_std*255} on test images...")
for batch, imagename in loaders['test']:
    batch = batch.cuda()
    noise = torch.randn_like(batch) * noise_std
    noisy_batch = batch + noise

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        output = model(noisy_batch)
        loss = (output - batch).pow(2).sum() / batch.shape[0]

    # statistics
    cur_mse = -10*np.log10(loss.item() / (batch.shape[2]*batch.shape[3]))
    print(f'{imagename[0]}:\t{cur_mse}')
    psnr += cur_mse
    num_iters += 1
print('===========================')
print(f'Average:\t{psnr/num_iters}')