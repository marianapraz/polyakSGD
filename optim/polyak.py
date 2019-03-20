import torch
from torch.optim.optimizer import Optimizer, required


class PolyakSGD(Optimizer):
    r"""Implements adaptive gradient descent using Polyak's time step.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        max_lr (float): maximum value the learning rate can take
        min_lr (float): minimum value the learning rate can take
        fstar (float): a "reasonable" approximation of the true minimum value of 
                        the true loss. For example, values around the minimum 
                        value of the loss of a validation/test set after a good 
                        baseline run. 
        window (int): the number of gradient steps between each learning rate 
                        update. For example, we have found 7500 steps is a 
                        good choice, which is roughly 20 epochs on the CIFAR datasets.

    Example:
        >>> optimizer = PolyakSGD(model.parameters(), max_lr = 0.1, 
                            fstar = 0.200,
                            window = 20*grad_steps_in_epoch)
        >>> optimizer.zero_grad()
        >>> loss = loss_fn(model(data), target)
        >>> runavg_loss = 0.9 *runavg_loss + 0.1*loss
        >>> loss.backward()
        >>> optimizer.step(runavg_loss)

    .. note::
        The optimizer requires a value for the current true training loss (*not* the batch loss).
        For smaller datasets this can be evaluated during test time after each epoch. 
        If this is not possible then a running exponential average of 
        the current epoch's batch losses might suffice. On CIFAR-10 both 
        options give good results, although the best fstar values are different.
    """

    def __init__(self, params, max_lr=required, min_lr = 0.00001, fstar=0.0, 
        window=7500):

        if max_lr is not required and max_lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if fstar < 0.0:
            raise ValueError("Invalid fstar: {}".format(fstar))
        if not isinstance(window,int):
            raise ValueError("Invalid window: {}".format(window))
        
        defaults = dict(min_lr=min_lr, max_lr=max_lr, lr=max_lr, sumsq=0.0, 
            count=0, totalcount=0)
        
        self.fstar = fstar
        self.window = window
        super(PolyakSGD, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(PolyakSGD, self).__setstate__(state)


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
                dt = max(min(dt, group['max_lr']), group['min_lr']) 
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
