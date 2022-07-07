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
from typing import Any

import numpy as np
from findiff import FinDiff
from scipy.integrate import solve_ivp

from lpde.dataset import Dataset
from lpde.utils import get_dudt_and_reshape_data


def ensure_boundary_conditions(values):
    """
    Ensure zero-flux boundary conditions.

    Args:
        values: numpy array containing variables

    Returns:
        Array with zero-flux boundaries
    """
    values[:, 0] = values[:, 1]
    values[:, -1] = values[:, -2]
    values[0, :] = values[1, :]
    values[-1, :] = values[-2, :]
    return values


def dudt(time: float,  # pylint: disable=unused-argument
         values: np.ndarray,
         length: float,
         c_1: float,
         c_2: float) -> np.ndarray:
    """
    Time derivative of the complex Ginzburg-Landau equation.

    Args:
        t: time step
        values: numpy array containing variables
        length: length of spatial domain
        c_1: parameter c1
        c_2: parameter c2

    Returns:
        Array with du/dt data
    """
    num_grid_points = int(np.sqrt(len(values)))
    delta_x = length/num_grid_points
    values = np.reshape(values, (num_grid_points, num_grid_points))

    dydxx = FinDiff(0, delta_x, 2)(values.real) + 1.0j*FinDiff(0, delta_x, 2)(values.imag) + \
        FinDiff(1, delta_x, 2)(values.real) + 1.0j * \
        FinDiff(1, delta_x, 2)(values.imag)
    dydt = values - (1+1.0j*c_2)*np.abs(values)**2*values + (1+1.0j*c_1)*dydxx
    return ensure_boundary_conditions(dydt).flatten()


def create_initial_conditions(n_grid_points: int) -> np.ndarray:
    """
    Specify initial conditions.

    Args:
        n_grid_points: number of spatial grid points

    Returns:
        Array with initial values
    """
    values = 0.5 * np.random.randn(n_grid_points, n_grid_points) + \
        0.5j * np.random.randn(n_grid_points, n_grid_points)
    return ensure_boundary_conditions(values).flatten()


def integrate(n_grid_points: int = 80,
              n_time_steps: int = 200,
              t_min: float = 20.0,
              t_max: float = 40.0,
              pars: Any = None):
    """
    Integrate complex Ginzburg-Landau equation.

    Args:
        n_grid_points: number of spatial grid points
        n_time_steps: number of time steps to sample data from
        t_min: start of time window
        t_max: end of time window
        pars: list of system parameters containing:
            length: length of spatial domain
            c_1: parameter c1
            c_2: parameter c2

    """
    # Default parameters if none are passed
    pars = [80, 0.0, -3.0] if pars is None else pars
    length, c_1, c_2 = pars

    # Write the parameters into a dictionary for future use.
    data_dict = {}
    data_dict['c_1'] = c_1
    data_dict['c_2'] = c_2
    data_dict['length'] = length
    data_dict['n_grid_points'] = n_grid_points
    data_dict['t_min'] = t_min
    data_dict['t_max'] = t_max
    data_dict['n_time_steps'] = n_time_steps

    # Set initial_conditions.
    initial_condition = create_initial_conditions(n_grid_points)
    data_dict['initial_condition'] = initial_condition

    # Set time vector.
    t_eval = np.linspace(t_min, t_max, n_time_steps+1, endpoint=True)

    data_dict['t_eval'] = t_eval-t_min

    # Compute solution.
    print('Computing the solution.')
    sol = solve_ivp(dudt,
                    [0, t_eval[-1]],
                    initial_condition,
                    t_eval=t_eval,
                    args=(length, c_1, c_2))
    data_dict['data'] = sol.y.T
    data_dict['data'] = data_dict['data'].reshape((data_dict['data'].shape[0],
                                                   n_grid_points,
                                                   n_grid_points))
    return data_dict


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
        pars = [self.config.getfloat('length'),
                self.config.getfloat('c_1'),
                self.config.getfloat('c_2')]
        data_dict = integrate(n_grid_points=self.config.getint('n_grid_points'),
                              n_time_steps=self.config.getint('n_time_steps'),
                              t_min=self.config.getfloat('tmin'),
                              t_max=self.config.getfloat('tmax'),
                              pars=pars)

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
        delta_y = np.repeat(data_dict['length'] /
                            data_dict['n_grid_points'], len(data))
        delta_xy = np.stack((delta_x, delta_y), axis=-1)

        x_data, delta_xy, y_data = get_dudt_and_reshape_data(data,
                                                             delta_xy,
                                                             delta_t,
                                                             self.config.getint('fd_dt_acc'))

        # Introduce variable dimension if it does not exist yet
        if len(x_data.shape) == 2:
            return x_data[:, np.newaxis], delta_x, y_data[:, np.newaxis]
        return np.transpose(x_data, (0, 3, 1, 2)), delta_xy, np.transpose(y_data, (0, 3, 1, 2))
