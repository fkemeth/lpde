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
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from lpde.dataset import Dataset
from lpde.utils import get_dudt_and_reshape_data


def dudt(time: float,  # pylint: disable=unused-argument
         values: np.ndarray,
         lambda_par: float,
         omega: np.ndarray,
         coupling_const: float) -> np.ndarray:
    """
    Time derivative of the complex Stuart-Landau ensemble.

    Args:
        t: time step
        values: numpy array containing variables
        lambda_par: parameter lambda
        omega: parameter omega
        coupling_const: coupling constant

    Returns:
        Array with du/dt data
    """
    return (lambda_par + 1.0j*omega)*values - np.abs(values)**2*values + \
        coupling_const*(np.mean(values)-values)


def create_initial_conditions(n_oscillators: int) -> np.ndarray:
    """
    Specify initial conditions.

    Args:
        n_oscillators: number of oscillators

    Returns:
        Array with initial values
    """
    return 0.5 * np.random.randn(int(n_oscillators)) + \
        0.5j * np.random.randn(int(n_oscillators))


def integrate(n_oscillators: int = 256,
              n_time_steps: int = 200,
              t_min: float = 1000.0,
              t_max: float = 1200.0,
              pars: Any = None):
    """
    Integrate complex Stuart-Landau ensemble.

    Args:
        n_oscillators: number of oscillators
        n_time_steps: number of time steps to sample data from
        t_min: start of time window
        t_max: end of time window
        pars: list of system parameters containing:
            lambda_par: parameter lambda
            gamma: width of omega interval
            omega_off: gamma offset
            coupling_const: coupling constant K

    """
    # Default parameters if none are passed
    pars = [1.0, 1.7, 0.2, 1.2] if pars is None else pars
    lambda_par, gamma, omega_off, coupling_const = pars

    # Write the parameters into a dictionary for future use.
    data_dict = {}
    data_dict['lambda_par'] = lambda_par
    data_dict['gamma'] = gamma
    data_dict['omega_off'] = omega_off
    data_dict['n_oscillators'] = n_oscillators
    data_dict['t_min'] = t_min
    data_dict['t_max'] = t_max
    data_dict['n_time_steps'] = n_time_steps

    omega = np.linspace(-gamma, gamma, n_oscillators)+omega_off
    data_dict['omega'] = omega

    # Set initial_conditions.
    initial_condition = create_initial_conditions(n_oscillators)
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
                    args=(lambda_par, omega, coupling_const))
    data_dict['data'] = sol.y.T
    return data_dict


class SLEDataset(Dataset):
    """
    SLE dataset.

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
        pars = [self.config.getfloat('lambda_par'),
                self.config.getfloat('gamma'),
                self.config.getfloat('omega_off'),
                self.config.getfloat('coupling_const')]

        data_dict = integrate(n_oscillators=self.config.getint('n_oscillators'),
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

        delta_x = np.repeat(2*data_dict['gamma'] /
                            data_dict['n_oscillators'], len(data))

        x_data, delta_x, y_data = get_dudt_and_reshape_data(data,
                                                            delta_x,
                                                            delta_t,
                                                            self.config.getint('fd_dt_acc'))

        self.t_eval = data_dict['t_eval'][
            int(self.config.getint('fd_dt_acc')/2):-int(self.config.getint('fd_dt_acc')/2)]
        self.t_eval -= self.t_eval[0]

        # Introduce variable dimension if it does not exist yet
        if len(x_data.shape) == 2:
            return x_data[:, np.newaxis], delta_x, y_data[:, np.newaxis]
        return np.transpose(x_data, (0, 2, 1)), delta_x, np.transpose(y_data, (0, 2, 1))

    def create_boundary_functions(self, boundary_width: float):
        """
        Create functions specifying the boundary dynamics.
        """
        left_boundary_data = self.x_data[:, :, :boundary_width]
        right_boundary_data = self.x_data[:, :, -boundary_width:]

        left_boundary_func = interp1d(self.t_eval, left_boundary_data, axis=0)
        right_boundary_func = interp1d(
            self.t_eval, right_boundary_data, axis=0)

        return (left_boundary_func, right_boundary_func)
