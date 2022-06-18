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

def progress(train_loss: float, val_loss: float) -> str:
    """
    Create progress bar description.

    Argss
        train_loss: Training loss
        val_loss: Validation or test loss

    Returns:
        String with training and test loss
    """
    return 'Train/Loss: {:.8f} ' \
           'Val/Loss: {:.8f}' \
           .format(train_loss, val_loss)


def get_dudt_finite_difference(x_data: np.ndarray,
                               delta_t: float,
                               fd_dt_acc: int) -> np.ndarray:
    """
    Calculate du/dt using finite differences.

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
    y_data = get_dudt_finite_difference(x_data, delta_t, fd_dt_acc)

    x_data = x_data[int(fd_dt_acc/2):-int(fd_dt_acc/2)]
    delta_x = delta_x[int(fd_dt_acc/2):-int(fd_dt_acc/2)]
    return x_data, delta_x, y_data
