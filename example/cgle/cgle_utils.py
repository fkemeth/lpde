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
from scipy.fftpack import diff as psdiff
from scipy.integrate import solve_ivp


def to_real(values: np.ndarray) -> np.ndarray:
    """
    Convert complex array to real array.

    Args:
        values: numpy array containing complex values

    Returns:
        Real array with shape 2*len(values)
    """
    return [*values.real, *values.imag]


def to_complex(values: np.ndarray) -> np.ndarray:
    """
    Convert complex array to real array.

    Args:
        values: numpy array containing real values

    Returns:
        Complex array with shape len(values)/2
    """
    return values[:int(len(values)/2)] + 1.0j*values[int(len(values)/2):]


def dudt(time: float,  # pylint: disable=unused-argument
         values: np.ndarray,
         length: float,
         c_1: float,
         c_2: float) -> np.ndarray:
    """
    Time derivative of the complex Ginzburg-Landau equation.

    Args:
        t: time step
        y: numpy array containing variables
        length: length of spatial domain
        c_1: parameter c1
        c_2: parameter c2

    Returns:
        Array with du/dt data
    """
    values = to_complex(values)
    dydxx = psdiff(values.real, period=length, order=2) + \
        1.0j*psdiff(values.imag, period=length, order=2)
    dydt = values - (1+1.0j*c_2)*np.abs(values)**2*values + (1+1.0j*c_1)*dydxx
    return to_real(dydt)


def create_initial_conditions(n_grid_points: int) -> np.ndarray:
    """
    Specify initial conditions.

    Args:
        n_grid_points: number of spatial grid points

    Returns:
        Array with initial values
    """
    return 0.5 * np.random.randn(int(2*n_grid_points))


def integrate(n_grid_points: int = 256,
              n_time_steps: int = 200,
              t_min: float = 1000.0,
              t_max: float = 1200.0,
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

    # Compute solution.
    print('Computing the solution.')
    sol = solve_ivp(dudt,
                    [0, t_eval[-1]],
                    initial_condition,
                    t_eval=t_eval,
                    args=(length, c_1, c_2))
    data_dict['data'] = sol.y.T
    data_dict['data'] = data_dict['data'][:, :n_grid_points] + \
        1.0j*data_dict['data'][:, n_grid_points:]
    return data_dict