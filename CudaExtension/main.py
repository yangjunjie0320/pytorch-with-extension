import torch, time
from torch.nn import Module
from torch.autograd import Function

def sigmoid_cuda_extension(x):
    import sigmoid_with_cuda_extension as sigmoid

    class SigmoidFunction(Function):
        @staticmethod
        def forward(ctx, x):
            y = sigmoid.forward(x)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, dy):
            dy = dy.contiguous()
            g = sigmoid.backward(*ctx.saved_tensors, dy)
            return g

    class Sigmoid(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return SigmoidFunction.apply(x)
    
    f = Sigmoid().cuda()
    return f(x)

def sigmoid_torch_func(x):
    return torch.exp(-x) / (1.0 + torch.exp(-x))

def sigmoid_for_loop(x):
    y = torch.zeros_like(x)
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            y[i, j] = torch.exp(-x[i, j]) / (1.0 + torch.exp(-x[i, j]))
    return y
    
if __name__ == "__main__":
    xx = torch.randn(200, 200).requires_grad_()
    yy = 1.0 - torch.sigmoid(xx)
    yy.sum().backward()
    gg = xx.grad

    for s in [sigmoid_cuda_extension, sigmoid_torch_func, sigmoid_for_loop]:
        x = xx.clone().detach().requires_grad_()
        if "cuda" in s.__name__:
            x = xx.clone().detach().cuda().requires_grad_()

        t0 = time.time()
        y = s(x)
        y.sum().backward()
        g = x.grad
        t = time.time() - t0
        
        err  = torch.norm(y.cpu() - yy).item()
        err += torch.norm(g.cpu() - gg).item()
        err /= yy.size(0) * yy.size(1)
        print(f"Time for {s.__name__:30s}: {t:.4e} s, err: {err:.4e}")

