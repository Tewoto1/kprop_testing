'''
I want to find the principal directions of the trained model at each hidden layer form a large batch of
random Gaussian inputs, along with their eigenvalues. I will pass the principal directions into the next layer
to see how their norm gets shrunk.
The results will be compared to an untrained checkpoint
We plot the eigenvalues of the top directions, and plot the norm shrink after the layer, compared to the untrained
baseline.
'''

import os

from .Tools import *
from ..model import *

models = []
model_names = []
dir = "checkpoints/trained_to_0_checkpoints"
for filename in os.listdir(dir):
    if filename.endswith(".pt"):
        model, _ = MLP.load(os.path.join(dir, filename))
        models.append(model)
        model_names.append(filename)