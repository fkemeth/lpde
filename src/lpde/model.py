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

###############################################################################
#                                                                             #
# Apr 2022                                                                    #
# felix@kemeth.de                                                             #
#                                                                             #
###############################################################################

from configparser import ConfigParser


import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import TruncatedSVD


class Model:
    """
    Wrapper around neural network.

    Includes functions to train and validate network.

    Arguments:
    dataloader_train          - Dataloader with training data
    dataloader_val            - Dataloader with validation or test data
    network                   - PyTorch module with the network topology
    config                    - Config with hyperparameters
    path                      - Path where the model should be saved
    """

    def __init__(self,
                 dataloader_train: DataLoader,
                 dataloader_val: DataLoader,
                 network: torch.nn.Module,
                 config: ConfigParser,
                 path: str = ''):
        super().__init__()
        self.base_path = path

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.net = network
        self.device = self.net.device
        print('Using:', self.device)
        self.net = self.net.to(self.device)

        self.learning_rate = float(config["lr"])

        self.criterion = torch.nn.MSELoss(reduction='sum').to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=int(config["patience"]),
            factor=float(config["reduce_factor"]), min_lr=1e-7)

    def train(self) -> float:
        """
        Train model over one epoch.

        Returns
        -------
        Loss averaged over the training data
        """
        self.net = self.net.train()

        sum_loss, cnt = 0, 0
        for (data, delta_x, target, param) in self.dataloader_train:
            data = data.to(self.device)
            delta_x = delta_x.to(self.device)
            target = target.to(self.device)
            if self.net.use_param:
                param = param.to(self.device)

            # backward
            self.optimizer.zero_grad()

            # forward
            if self.net.use_param:
                output = self.net(data, delta_x, param)
            else:
                output = self.net(data, delta_x)

            # compute loss
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            # measure accuracy on batch
            sum_loss += loss
            cnt += 1

        return sum_loss / cnt

    def validate(self) -> float:
        """
        Validate model on validation set.

        Updates learning rate using scheduler.

        Updates best accuracy.

        Returns
        -------
        Loss averaged over the validation data
        """
        self.net = self.net.eval()

        sum_loss, cnt = 0, 0
        with torch.no_grad():
            for (data, delta_x, target, param) in self.dataloader_val:
                data = data.to(self.device)
                delta_x = delta_x.to(self.device)
                target = target.to(self.device)
                if self.net.use_param:
                    param = param.to(self.device)

                # forward
                if self.net.use_param:
                    output = self.net(data, delta_x, param)
                else:
                    output = self.net(data, delta_x)

                # loss / accuracy
                sum_loss += self.criterion(output, target)
                cnt += 1

        # Learning Rate reduction
        self.scheduler.step(sum_loss / cnt)

        return sum_loss / cnt

    def save_network(self, name: str) -> str:
        """
        Save model to disk.

        Arguments
        -------
        name         - Model filename.

        Returns
        -------
        Model filename.
        """
        model_file_name = self.base_path+name
        torch.save(self.net.state_dict(), model_file_name)
        return name

    def load_network(self, name: str) -> None:
        """
        Load model from disk.

        Arguments
        -------
        name         - Model filename.
        """
        model_file_name = self.base_path+name
        self.net.load_state_dict(torch.load(model_file_name))

    def integrate_svd(self,
                      dataset: Dataset,
                      svd: TruncatedSVD,
                      idx: int,
                      horizon: int) -> np.ndarray:
        """
        Integrate idx'th snapshot of dataset for horizon time steps using Euler stepper.

        Arguments:
        dataset                   - Dataset containing snapshots
        svd                       - Truncated SVD for regulairzation
        idx                       - Index of initial snapshot
        horizon                   - Number of time steps to integrate forward

        Returns
        -------
        Numpy array with integrated data
        """
        left_bounds, _, right_bounds, _, _, param = dataset.get_data(True)
        data = []
        data0 = svd.inverse_transform(
            svd.transform(dataset.x_data[idx].reshape(1, -1)))
        data.append(data0.reshape(2, -1))

        for i in range(idx, horizon+idx):
            pred_f = self.net.forward(
                torch.tensor(data[-1], dtype=torch.get_default_dtype()
                             ).unsqueeze(0).to(self.net.device),
                dataset.__getitem__(i)[1].unsqueeze(0).to(
                    self.net.device),
                torch.tensor(param[idx], dtype=torch.get_default_dtype()
                             ).unsqueeze(0).to(self.net.device)
            )[0].cpu().detach().numpy()
            prediction = data[-1][:, dataset.off_set:-dataset.off_set] + dataset.delta_t*pred_f

            prediction = np.concatenate((left_bounds[i+1], prediction, right_bounds[i+1]), axis=1)
            prediction = svd.inverse_transform(
                svd.transform(prediction.reshape(1, -1)))
            data.append(prediction.reshape(2, -1))
        return np.array(data)
