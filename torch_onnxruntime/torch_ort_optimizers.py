import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
import torch_ort

class SGD(Optimizer):
    r"""Implements stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            torch_ort.optimizers.SGD(group['params'], group['lr'])

        return loss

class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay_mode (int, optional): weight decay mode
        Currently two modes of Adamw are supported:
        Mode 0: Pytorch https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW,
                 bias correction is applied on m and v individually,
                 weight decay is applied before weight is updated.
        Mode 1: Huggingface https://huggingface.co/transformers/_modules/transformers/optimization.html#AdamW.,
                 bias correction is applied on learning rate,
                 weight decay is applied after weight is updated.        
        do_bias_correction (int, optional): whether to perform bias correction based on steps.
            (default: 1 (True))
    """

    def __init__(self, params, lr=1e-3, alpha=0.9, beta=0.999, lam=0.0, eps=1e-8,
                 weight_decay_mode=0, do_bias_correction=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha < 1.0:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta value: {}".format(beta))
        if not 0.0 <= lam:
            raise ValueError("Invalid lambda value: {}".format(lam))
        if weight_decay_mode not in (0, 1):
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, alpha=alpha, beta=beta, lam=lam, eps=eps,
                        weight_decay_mode=weight_decay_mode, do_bias_correction=do_bias_correction)
        super(AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = torch.LongTensor([0])
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, device=p.device)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, device=p.device)          

            torch_ort.optimizers.Adam(group['lr'], group['alpha'], group['beta'], group['lam'],
            group['eps'], group['weight_decay_mode'], group['do_bias_correction'], p, state['step'], 
            state['exp_avg'], state['exp_avg_sq'])
        return loss
