"""
Copyright © 2022 Felix P. Kemeth

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from dataset import Dataset

import lpde


def main(config: ConfigParser):
    """
    Train a partial differential equation on simulation data.

    Args:
       config: config with hyperparameters
    """

    # Create Dataset
    dataset_train = Dataset(config['SYSTEM'])
    dataset_test = Dataset(config['SYSTEM'])

    # Create Dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config['TRAINING'].getint('batch_size'), shuffle=True,
        num_workers=config['TRAINING'].getint('num_workers'), pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config['TRAINING'].getint('batch_size'), shuffle=False,
        num_workers=config['TRAINING'].getint('num_workers'), pin_memory=True)

    # Create the network architecture
    network = lpde.network.Network(
        config['MODEL'], n_vars=dataset_train.x_data.shape[1])

    # Create a model wrapper around the network architecture
    # Contains functions for training
    model = lpde.model.Model(
        dataloader_train, dataloader_test, network, config['TRAINING'])

    progress_bar = tqdm.tqdm(range(0, config['TRAINING'].getint('epochs')),
                             total=config['TRAINING'].getint('epochs'),
                             leave=True, desc=lpde.utils.progress(0, 0))

    # Train the model
    train_loss_list = []
    val_loss_list = []
    for _ in progress_bar:
        train_loss = model.train()
        val_loss = model.validate()
        progress_bar.set_description(lpde.utils.progress(train_loss, val_loss))

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    t_eval = np.linspace(0,
                         config['SYSTEM'].getfloat(
                             'tmax')-config['SYSTEM'].getfloat('tmin'),
                         config['SYSTEM'].getint('n_time_steps')+1, endpoint=True)
    initial_condition, delta_x, _ = dataset_test[0]
    _, predictions = model.integrate(initial_condition.detach().numpy(),
                                     [delta_x.detach().numpy()],
                                     t_eval=t_eval)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.pcolor(dataset_test.x_data[:, 0])
    ax2 = fig.add_subplot(122)
    ax2.pcolor(predictions[:, 0])
    plt.show()


if __name__ == '__main__':
    cfg = ConfigParser()
    cfg.read('config.cfg')
    main(cfg)
