import deepchem
print(deepchem.__version__)
import deepchem.models.torch_models.layers as torch_layers
print(dir(torch_layers))

from deepchem.models.torch_models.layers import GraphConv