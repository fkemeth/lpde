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
import numpy as np
from cgle_utils import integrate

from lpde.dataset import Dataset


def get_dudt(x_data: np.ndarray,
             delta_t: float,
             fd_dt_acc: int) -> np.ndarray:
    """
    Calculate du/dt.

    Args:
        x_data: Array with snapshot data
        delta_t: Float with dt between snapshots
        fd_dt_acc: Int specifying finite difference order

    Returns:
        Array with du/dt data
    """
    # Approximate du/dt using finite differences
    if fd_dt_acc == 2:
        # accuracy 2
        y_data = (x_data[2:]-x_data[:-2])/(2*delta_t)
    elif fd_dt_acc == 4:
        # accuracy 4
        y_data = (x_data[:-4]-8*x_data[1:-3]+8 *
                  x_data[3:-1]-x_data[4:])/(12*delta_t)
    else:
        raise ValueError(
            'Finite difference in time accuracy must be 2 or 4.')
    return y_data


def get_dudt_and_reshape_data(x_data: np.ndarray,
                              delta_x: np.ndarray,
                              delta_t: float,
                              fd_dt_acc: int):
    """
    Calculate du/dt and reshape data.

    Args:
        x_data: Array with snapshot data
        delta_x: Array with delta_x data
        delta_t: Float with dt between snapshots
        fd_dt_acc: Int specifying finite difference order

    Returns:
        Array with snapshot data
        Array with delta_x data
        Array with du/dt data
    """
    # Approximate du/dt using finite differences
    y_data = get_dudt(x_data, delta_t, fd_dt_acc)

    x_data = x_data[int(fd_dt_acc/2):-int(fd_dt_acc/2)]
    delta_x = delta_x[int(fd_dt_acc/2):-int(fd_dt_acc/2)]
    return x_data, delta_x, y_data


class CGLEDataset(Dataset):
    """
    CGLE dataset.

    Args:
        config: configfile with dataset parameters
    """

    def create_data(self) -> list:
        """
        Create partial differential equation dataset.

        Returns:
            Array with snapshot data
            Array with delta_x data
            Array with du/dt data
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

        delta_x = np.repeat(data_dict['length'] /
                            data_dict['n_grid_points'], len(data))

        x_data, delta_x, y_data = get_dudt_and_reshape_data(data,
                                                            delta_x,
                                                            delta_t,
                                                            self.config.getint('fd_dt_acc'))

        # Introduce variable dimension if it does not exist yet
        if len(x_data.shape) == 2:
            return x_data[:, np.newaxis], delta_x, y_data[:, np.newaxis]
        return np.transpose(x_data, (0, 2, 1)), delta_x, np.transpose(y_data, (0, 2, 1))
