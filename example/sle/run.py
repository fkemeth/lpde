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
import torch
import tqdm
from dataset import SLEDataset

import lpde


def main(config: ConfigParser):  # pylint: disable-msg=too-many-locals
    """
    Train a partial differential equation on simulation data.

    Args:
       config: config with hyperparameters
    """
    # Use cuda only if available
    config['MODEL']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create Dataset
    boundary_width = int((config['MODEL'].getint('kernel_size')-1)/2)
    dataset_train = SLEDataset(config['SYSTEM'], boundary_width=boundary_width)
    dataset_test = SLEDataset(config['SYSTEM'], boundary_width=boundary_width)

    # Visualize training data
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    pl1 = ax1.pcolor(dataset_train.x_data[::5, 0], cmap='plasma')
    ax1.set_xlabel(r'$x_i$')
    ax1.set_ylabel(r'$t_i$')
    plt.colorbar(pl1, label=r'Re $W$')
    plt.tight_layout()
    plt.savefig('./fig/training_data.png')
    plt.show()

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

    # Make predictions
    t_eval = dataset_test.t_eval[:10]
    initial_condition, delta_x, _ = dataset_test[0]
    _, predictions = model.integrate(
        initial_condition.detach().numpy()[:, boundary_width:-boundary_width],
        [delta_x.detach().numpy()],
        t_eval=t_eval,
        boundary_functions=dataset_test.boundary_functions)

    # Visualize training data
    fig = plt.figure(figsize=(8, 3.6))
    ax1 = fig.add_subplot(121)
    pl1 = ax1.pcolor(dataset_test.x_data[::5, 0], cmap='plasma')
    ax1.set_xlabel(r'$x_i$')
    ax1.set_ylabel(r'$t_i$')
    ax1.set_title('test data')
    plt.colorbar(pl1, label=r'Re $W$')
    ax2 = fig.add_subplot(122)
    pl2 = ax2.pcolor(predictions[::5, 0], cmap='plasma')
    ax2.set_xlabel(r'$x_i$')
    ax2.set_ylabel(r'$t_i$')
    ax2.set_title('prediction')
    plt.colorbar(pl2, label=r'prediction Re $W$')
    plt.tight_layout()
    plt.savefig('./fig/test_data_and_prediction.png')
    plt.show()


if __name__ == '__main__':
    cfg = ConfigParser()
    cfg.read('config.cfg')
    main(cfg)
