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

import findiff
import torch

from lpde import network


def get_default_model_config() -> configparser.ConfigParser:
    """
    Create default model config.

    Returns:
        Config with default values
    """
    config = configparser.ConfigParser()
    config['MODEL'] = {'kernel_size': '5',
                       'device': 'cuda',
                       'use_param': 'False',
                       'num_params': '2',
                       'n_filters': '16',
                       'n_layers': '2',
                       'n_derivs': '2'}
    return config


def test_coefficients():
    """
    Test finite difference coefficients.
    """
    config = get_default_model_config()

    for kernel_size in ['3', '5', '9']:
        for n_var in [1, 2, 3, 4]:
            for n_derivs in [str(x) for x in range(1, int(kernel_size))]:
                config['MODEL']['kernel_size'] = kernel_size
                config['MODEL']['n_derivs'] = n_derivs
                net = network.Network(config['MODEL'], n_var)

                print(net.coeffs.shape)

                assert net.coeffs.shape == (int(n_derivs)+1, 1, int(kernel_size)), \
                    'Coefficients should be of shape (n_derivs+1, n_var, kernel_size).'
                assert net.get_coeffs(min_deriv=0, max_deriv=int(n_derivs)).shape == \
                    (int(n_derivs)+1, 1, int(kernel_size)), \
                    'Coefficients should be of shape (n_derivs+1, n_var, kernel_size).'

                zero_deriv = torch.zeros(int(kernel_size)).to(net.device)
                zero_deriv[int((int(kernel_size)-1)/2)] = 1

                assert torch.equal(net.coeffs[0, 0], zero_deriv), \
                    'Zeroth order derivative is wrong.'

                if int(kernel_size) == 3 and int(n_derivs) > 1:
                    first_deriv = torch.tensor([-0.5, 0, 0.5],
                                               dtype=torch.get_default_dtype()).to(net.device)
                    assert torch.equal(net.coeffs[1, 0], first_deriv), \
                        'First order derivative is wrong.'
                    second_deriv = torch.tensor([1, -2, 1],
                                                dtype=torch.get_default_dtype()).to(net.device)
                    assert torch.equal(net.coeffs[2, 0], second_deriv), \
                        'Second derivative is wrong.'

                for deriv_order in range(1, int(n_derivs)):
                    acc_order = 0
                    fd_coeff = []
                    while len(fd_coeff) < int(kernel_size):
                        acc_order += 2
                        fd_coeff = findiff.coefficients(deriv_order,
                                                        acc_order)['center']['coefficients']

                    ith_deriv = torch.tensor(fd_coeff,
                                             dtype=torch.get_default_dtype()).to(net.device)

                    assert torch.equal(net.coeffs[int(deriv_order), 0], ith_deriv), \
                        str(deriv_order)+' order derivative is wrong.'
