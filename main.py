import torch, time
from torch.nn import Module
from torch.autograd import Function

import mysigmoid

class MySigmoidFunction(Function):
    @staticmethod
    def forward(ctx, x):
        y = mysigmoid.forward(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        g = mysigmoid.backward(*ctx.saved_tensors, grad_output)
        return g


class MySigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return MySigmoidFunction.apply(x)
    
if __name__ == "__main__":
    x1 = torch.arange(2, 12).view(2, 5).float().requires_grad_()
    x2 = torch.arange(2, 12).view(2, 5).float().requires_grad_()
    
    t0 = time.time()
    func = MySigmoid()
    fx1 = func(x1)
    fx1.sum().backward()
    t = time.time() - t0
    print(f"Time for MySigmoid:     {t:.4e} s")

    t0 = time.time()
    fx2 = torch.exp(-x2) / (torch.exp(-x2) + 1.0)
    fx2.sum().backward()
    t = time.time() - t0
    print(f"Time for torch.sigmoid: {t:.4e} s")

    assert torch.allclose(fx1, fx2)
    assert torch.allclose(x1.grad, x2.grad)
