# Learning Partial Differential Equations

[![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://badge.fury.io/py/lpde)
[![PyPI version](https://badge.fury.io/py/lpde.svg)](https://badge.fury.io/py/lpde)
![PyPI - License](https://img.shields.io/pypi/l/lpde)
[![Python package](https://github.com/fkemeth/lpde/actions/workflows/python-package.yml/badge.svg)](https://github.com/fkemeth/lpde/actions/workflows/python-package.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/lpde.svg)]()
[![Downloads](https://static.pepy.tech/personalized-badge/lpde?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/lpde)

INSTALLATION
---------


By way of pip:

`pip install lpde`

By way of source

    git clone https://github.com/fkemeth/lpde
    cd lpde
    pip install .

USAGE
---------

This python package contains functions to learn partial differential equations (PDE) from data.


In order to learn a PDE on a set of training data that can be used for prediction, several parameters
have to be specified. These can be defined in a `config.cfg` file, which is read using a config parser. The following hyper parameters can be specified:

- Under the subsection `SYSTEM`, parameters used for generating the data are defined.
  In the example considered here, these included

  - `n_time_steps`: The number of time steps at which training data is collected.
  - `n_grid_points`: The number of spatial grid points at which training data is collected.
  - `length`: The length of the one-dimensional spatial interval. Together with `n_grid_points`, this defines the spatial resolution `delta_x` that is also used to calculate the partial derivatives.
  - `tmin`: The time step above which training and test data is collected.
  - `tmax`: The time step until which training and test data is collected. Together with `tmin` and `n_time_steps`, this is used to calculate the temporal resolution `delta_t`.
  - `use_fd_dt`: If the time derivative of the variables at each point in space and time shall be calculated using finite differences.
  - `fd_dt_acc`: The accuracy order of the finite differences for computing the time derivative of the variables.

- Under the subsection `MODEL`, parameters specifying the neural network PDE are defined.
  These parameter are used to create and object of the `Network` class, and include

  - `kernel_size`: The width of the finite difference stencil used to calculate input spatial derivatives.
  - `device`: If to use either 'cpu' or 'cuda.
  - `use_param`: Boolean that specifies if to use additional parameters as input to the PDE. This is required if one wants to do bifurcation analysis of the learned PDE model.
  - `num_params`: If `use_param` is True, then here the number of additional system parameters have to be specified.
  - `n_filters`: The number of neurons in each layer of the PDE model.
  - `n_layers`: The number of layers of the PDE model.
  - `n_derivs`: The number of derivatives used in the PDE model.

- Under the subsection `TRAINING`, hyper parameters used for training the neural network PDE are specified. These are used to create an object of the `Model` class, a wrapper around the `Network` object, and include

  - `batch_size`: Batch size used for training.
  - `lr`: Initial learning rate used for training.
  - `weight_decay`: Strength of the L2 regularization applied to the neural network weights.
  - `epochs`: Number of epochs/tranining iterations.
  - `reduce_factor`: For reduced-learning-rate-on-plateau scheduler. Factor by which to reduce the learning rate.
  - `patience`: For reduced-learning-rate-on-plateau scheduler. Number of epochs to wait before reducing learning rate.

EXAMPLE
---------

The usage is best illustrated on an example.
Here, we show this on simulation data of an actual PDE, the complex Ginzburg-Landau equation,
with spatio-temporally chaotic dynamics, which is solved numerically on a one-dimensional
periodic domain.

This example can be run by using

    cd example/cgle/
    python run.py

The training data thereby looks like the data shown in the figure below.

![Training data](./example/cgle/fig/training_data.png)

Using the hyperparameters defined in `config.cfg`, a neural network PDE is learned on the data shown in the figure above, by optimizing its weights using backprobagation and the PyTorch framework.

The trained neural network PDE can then be used to make predictions on test data.
This is shown in the figure below, where on the left the actual test data is shown, and on the right
the predicted data is shown, obtained by integrating an initial snapshot at `t=0` forward in time
using the learned PDE model.

![Test data and predictions](./example/cgle/fig/test_data_and_prediction.png)


See [this GitHub repository](https://github.com/fkemeth/emergent_pdes) for further example usages.

ISSUES
---------

For questions, please contact (<felix@kemeth.de>), or visit [the GitHub repository](https://github.com/fkemeth/lpde).

LICENCE
---------

This work is licenced under MIT License.
Please cite

"Learning emergent partial differential equations
in a learned emergent space"
F.P. Kemeth et al.
*Nature Communications* 13, Article number: 3318 (2022)
(https://www.nature.com/articles/s41467-022-30628-6)

if you use this package for publications.
