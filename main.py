import torch, time
from torch.nn import Module
from torch.autograd import Function

import mysigmoid

class MySigmoidFunction(Function):
    @staticmethod
    def forward(ctx, input):
        y = mysigmoid.forward(input)
        ctx.save_for_backward(y)
        return mysigmoid.forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        g = mysigmoid.backward(*ctx.saved_tensors, grad_output)
        return g
    
class MySigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return MySigmoidFunction.apply(input)
    
if __name__ == "__main__":
    x1 = torch.arange(2, 12).view(2, 5).float().requires_grad_()

    func = MySigmoid()
    fx1 = func(x1)
    fx1.sum().backward()

    x2 = torch.arange(2, 12).view(2, 5).float().requires_grad_()
    fx2 = torch.exp(-x2) / (torch.exp(-x2) + 1)
    fx2.sum().backward()

    assert torch.allclose(fx1, fx2)
    assert torch.allclose(x1.grad, x2.grad)
