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

import configparser

import numpy as np

import findiff
import torch
import tqdm

import lpde

from example.cgle.dataset import CGLEDataset


def test_cgle_example():
    """
    Test using the CGLE example and train for one epoch.
    """
    config = configparser.ConfigParser()
    config.read('example/cgle/config.cfg')

    config["TRAINING"]["epochs"] = '2'
    config["MODEL"]["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create Dataset
    dataset_train = CGLEDataset(config['SYSTEM'])
    dataset_test = CGLEDataset(config['SYSTEM'])

    # Create Dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config['TRAINING'].getint('batch_size'), shuffle=True,
        pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config['TRAINING'].getint('batch_size'), shuffle=False,
        pin_memory=True)

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

    assert train_loss_list[-1]<train_loss_list[0], \
        'Training loss of CGLE example should decrease.'
