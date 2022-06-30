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

import torch


class Dataset(torch.utils.data.Dataset):
    """
    Partial differential equation dataset.

    Args:
        config: configfile with dataset parameters
    """

    def __init__(self, config: SectionProxy, boundary_width: int = 0) -> None:
        self.config = config
        self.x_data, self.delta_x, self.y_data = self.create_data()

        self.boundary_conditions = config['boundary_conditions']
        self.spatial_dimensions = self.x_data.shape[2:]
        self.num_spatial_dimensions = len(self.spatial_dimensions)

        if self.boundary_conditions == 'functional':
            self.boundary_functions = self.create_boundary_functions(
                boundary_width)

    def create_data(self) -> list:
        """
        Create partial differential equation dataset.

        Returns:
            Array with snapshot data
            Array with delta_x data
            Array with du/dt data
        """
        raise NotImplementedError(
            'You need to implement this method in %s.' % self.__class__.__name__)

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
