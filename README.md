This is a repository for Adaptive SGD. Adaptive SGD adapts Polyak's step size to stochastic gradients.

The repository is divided into several parts:
  * The folder `optim/` contains the optimizer code for AdaptiveSGD.
  * The `example.py` script contains an example for training and testing LeNet model for MNIST. The implementation of LeNet is in `models/` and the data is imported via the `dataloader.py` script. is contained in `experiments/classification`. 

## Optimizing using Polyak's step size for stochastic gradients
Import the optimizer from `optim/` and use the usual script from pytorch.

```python
from torch import nn
from models import LeNet
from optim import Adaptive

model = LeNet()
loss_fn = nn.CrossEntropy()
optimizer = Adaptive(model.parameters())

# Get training data x, labels y ...

yhat = model(x)
loss = loss_fn(yhat,y) 

# Now run .backward(), update the model, etc ... and
optimizer.step(runavg_loss)
# Define this runavg_loss depending on the problem. Could be replaced by a true loss in case of need.
```

