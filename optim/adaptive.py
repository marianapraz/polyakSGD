import torch
from torch.optim.optimizer import Optimizer, required


class Adaptive(Optimizer):
    r"""Implements adaptive gradient descent using Polyak's time step.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        max_lr (float): maximum value the learning rate can take
        fstar (float): a "reasonable" approximation of the true minimum value of 
                        the true loss. For example, values around the minimum 
                        value of the loss of a validation/test set after a good 
                        baseline run. 
        window (int): the number of gradient steps between each learning rate 
                        update. For example, the equivalent to 20 epochs is a 
                        good choice.

    Example:
        >>> optimizer = Adaptive(model.parameters(), max_lr = 0.1, 
                            fstar = 0.200,
                            window = 20*grad_steps_in_epoch)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(data), target).backward()
        >>> optimizer.step(runavg_loss)

    .. note::
        The optimizer requires an approximate value of the current training loss.
        For smaller datasets this can be evaluated. If not, a running average of 
        the current epochs batch losses might suffice. For test in cifar10 both 
        options got good results, although the best fstar value are different.
    """

    def __init__(self, params, max_lr=required, fstar=0.0, 
        window=10000):

        if max_lr is not required and max_lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if fstar < 0.0:
            raise ValueError("Invalid fstar: {}".format(fstar))
        if not isinstance(window,int):
            raise ValueError("Invalid window: {}".format(window))
        
        defaults = dict(max_lr=max_lr, lr=max_lr, sumsq=0.0, 
            count=0, totalcount=0)
        
        self.fstar = fstar
        self.window = window
        super(Adaptive, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(Adaptive, self).__setstate__(state)


    def step(self,runavg_loss):
        for group in self.param_groups:
            ## Store params' sum of gradients squared and update params
            p_sumsq = 0.0
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # Update the parameters sum of gradients squared
                grad = d_p.view(-1)
                p_sumsq += (grad.dot(grad)).item()
                # SGD step using the last lr defined
                p.data.add_(- group['lr'], d_p)

            ## Update the learning rate according to the window value
            if group['totalcount'] % self.window == 0 and group['totalcount']>0:
                dt = 2*(runavg_loss - self.fstar)/group['sumsq']
                dt = max(min(dt, group['max_lr']), 0.00001) 
                group['count'] = 0
                group['sumsq'] = 0.0
                print("\n[New learning rate]: %.2g\n"%dt)
                group['lr'] = dt

            # Update counters and group's mean of squared gradients 
            group['totalcount'] += 1
            group['count'] += 1
            delta = p_sumsq - group['sumsq']
            group['sumsq'] += 1/group['count']*delta
        
        return runavg_loss