## Optimizing using Polyak's step size or stochastic gradients

The basic update rule is 
` h_k = 2 \frac{f(x_k)- fstar}{\E{||g_k||^2}}. `

Here `f` represents the loss function ans `fstar` the value of `f` at the minimum. `g_k` are the minibatch gradients.
For details of the proof email me.

