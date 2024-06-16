import torch
from torch.optim import Optimizer
import torch.nn as nn
import torch.optim as optim


class Ruszczynski(Optimizer):
    r"""Implements Ruszczynski algorithm.

    It has been proposed in `Convergence of a stochastic subgradient method with
averaging for nonsmooth nonconvex constrained
optimization`_.
    we reformulated the constants so that to allow learning rate decay


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        mu : act as a momentum on the gradients (default: 0.90) it is a combined argument for a and t_k in the paper
    .. _Convergence of a stochastic subgradient method with averaging for nonsmooth nonconvex constrained optimization

    """

    def __init__(self, params, lr=1e-3, mu=0.9):
        if not 0 <= lr <= 1:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= mu <=1:
            raise ValueError("Invalid a value: {}".format(a))
        defaults = dict(lr=lr,mu=mu)
        super(Ruszczynski, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ruszczynski, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            approx_d_k = []


            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Ruszczynski does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['approx_d_k'] = p.grad.clone()
                        # Exponential moving average of squared gradient values
                        
                    approx_d_k.append(state['approx_d_k'])

                   
                    state['step'] += 1


            ruszczynski(params_with_grad,
                   grads,
                   approx_d_k,
                   beta=1,
                   a=group['mu']/group['lr'],
                   lr=group['lr'])
        return loss

def ruszczynski(params_with_grad,grads,approx_d_k,beta,a,lr):
    """
    Performs the inplace computation for the ruszczynski algorithm
    """
    for i, param in enumerate(params_with_grad):
        #xhi = (beta*param - approx_d_k[i])/beta
        param.add_(approx_d_k[i],alpha=-lr/beta)
        approx_d_k[i].mul_(1-a*lr).add_(grads[i],alpha=a*lr)

if __name__ == "__main__":
    # Testing the algorithm on a convex problem
    model = nn.Sequential(nn.Linear(2,1000),nn.ReLU(),nn.Linear(1000,1))
    X = torch.randn(10,2)/5
    m = torch.rand(2)
    Y = ((X-0.1)*m)**2
    optimizer=Ruszczynski(model.parameters(), lr=1e-3, a=300,beta=2)
    scheduler = optim.lr_scheduler.StepLR(optimizer,3000)
    # optimizer=torch.optim.Adam(model.parameters())
    for i in range(10000):
        loss = torch.sum((model(X)-Y)**2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        print(loss.item())
