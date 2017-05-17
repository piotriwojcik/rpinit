# rpinit
This repository hosts the Torch7 code for random projection initialization of neural networks from ESANN-2017 paper
"Random projection initialization for deep neural networks". Currently supported modules: nn.SpatialConvolution /
cudnn.SpatialConvolution.

Readme contents:

- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)


## Installation

See instructions on: https://luarocks.org/modules/piotriwojcik/rpinit.

## Usage

**`rpinit`** adds an `rpinitWeights` method to `nn.Module`, with the following API:

```lua
module:rpinitWeights(scheme, ...)
```

The [`scheme`] argument defines the initialization scheme: `kaiming`, `rp_gauss`, `rp_sparse`, `rp_achl`, `rp_count` or
`rp_hada`. `...` represents additional arguments for the initialiser function.
`rpinitWeights` method returns the module, allowing calls to be chained.


## Example

```lua
require 'nn'
require 'rpinit'
local nninit = require 'nninit'

layer1 = nn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
  :rpinitWeights('rp_count')

layer2 = nn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
  :rpinitWeights('rp_hada', 1000)     -- number of training examples: 1000; required by rp_hada scheme
  :init('bias', nninit.constant, 0)   -- additionally initialize biases to zeros
```
