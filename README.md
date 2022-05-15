# Learning Partial Differential Equations

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

- The main components consists of a neural network PDE class `Network(torch.nn.Module)`.
  To create an instance of this class, one needs to pass a config dictionary that specifies

  - `kernel_size`: The width of the finite difference stencil used to calculate input spatial derivatives
  - `n_derivs`: The number of derivatives used in the PDE model
  - `device`: Either 'cpu' or 'cuda
  - `use_param`: Boolean that specifies
  - `num_params`: If `use_param` is True, then here the number of parameters that change have to be specified.
  - `n_filters`: The number of neurons in each layer of the PDE model.
  - `n_layers`: The number of layers of the PDE model.

  In addition, the number of system variables `n_var` has to be provided

- Furthermore, a model wrapper to train and evaluate the neural network PDE is provided as a `Model` class. To create an instance of this class, one needs to provide

  - `dataloader_train`: A pytorch dataloader with the training data
  - `dataloader_val`: A pytorch dataloader with the validation data
  - `network`: A Network instance, as described above
  - `config`: A config dictionary containing
    - `lr`: The initial learning rate
    - `patience`: The patience used for the learning rate scheduler
    - `reduce_factor`: - The factor by which the learning rate is reduced when loss does not decrease
    - `weight_decay`: - Weight decay factor for regularization
  - `path`: String to the directory where the trained model shall be saved


See [this GitHub repository](https://github.com/fkemeth/emergent_pdes) for example usages.

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
(https://arxiv.org/abs/2012.12738)

if you use this package for publications.
