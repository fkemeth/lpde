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
from configparser import SectionProxy

import numpy as np
import torch
from cgle_utils import integrate


class Dataset(torch.utils.data.Dataset):
    """
    Partial differential equation dataset.

    Args:
        config: configfile with dataset parameters
    """

    def __init__(self, config: SectionProxy) -> None:
        self.config = config
        self.x_data, self.delta_x, self.y_data = self.create_data()

        self.boundary_conditions = 'periodic'

    def create_data(self) -> list:
        """
        Create partial differential equation dataset.

        Returns:
            Array with snapshot data
            Array with du/dt data
            Array with delta_x data
        """
        # Integrate pde
        data_dict = integrate(n_grid_points=self.config.getint('n_grid_points'),
                              n_time_steps=self.config.getint('n_time_steps'),
                              t_min=self.config.getfloat('tmin'),
                              t_max=self.config.getfloat('tmax'))

        # If data type is complex, transform to real data
        if data_dict['data'].dtype == 'complex':
            data = np.stack(
                (data_dict['data'].real, data_dict['data'].imag), axis=-1)
        else:
            data = data_dict['data']

        delta_t = (self.config.getfloat('tmax')-self.config.getfloat('tmin')) / \
            self.config.getint('n_time_steps')

        # Approximate du/dt using finite differences
        if self.config.getint('fd_dt_acc') == 2:
            # accuracy 2
            y_data = (data[2:]-data[:-2])/(2*delta_t)
            x_data = data[1:-1]
            delta_x = np.repeat(
                data_dict['length']/data_dict['n_grid_points'], len(data)-2)
        elif self.config.getint('fd_dt_acc') == 4:
            # accuracy 4
            y_data = (data[:-4]-8*data[1:-3]+8 *
                      data[3:-1]-data[4:])/(12*delta_t)
            x_data = data[2:-2]
            delta_x = np.repeat(
                data_dict['length']/data_dict['n_grid_points'], len(data)-4)
        else:
            raise ValueError(
                'Finite difference in time accuracy must be 2 or 4.')

        # Introduce variable dimension if it does not exist yet
        if len(x_data.shape) == 2:
            return x_data[:, np.newaxis], delta_x, y_data[:, np.newaxis]
        return np.transpose(x_data, (0, 2, 1)), delta_x, np.transpose(y_data, (0, 2, 1))

    def __len__(self) -> int:
        """
        Get length of dataset.

        Returns:
            Length of dataset.
        """
        return len(self.x_data)

    def __getitem__(self, index: int) -> tuple:
        """
        Get datapoint.

        Args:
            index: index of datapoint

        Returns:
            Tuple of input snapshot, delta_x and dudt.
        """

        _x = torch.tensor(self.x_data[index], dtype=torch.get_default_dtype())
        _dx = torch.tensor(self.delta_x[index],
                           dtype=torch.get_default_dtype())
        _y = torch.tensor(self.y_data[index], dtype=torch.get_default_dtype())
        return (_x, _dx, _y)
