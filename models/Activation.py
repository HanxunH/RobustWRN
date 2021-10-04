import torch.nn as nn
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return mish(input)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Siren(nn.Module):
    def forward(self, x):
        return torch.sin(30 * x)


class custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))

    def forward(self, x):
        min_x = torch.clamp(self.alpha*x, max=0)
        max_x = torch.clamp(self.beta*x, min=0)
        return (-1) * min_x + max_x


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


avaliable_activations = {"ReLU": nn.ReLU,
                         "Swish": MemoryEfficientSwish,
                         "Mish": Mish,
                         "GELU": nn.GELU,
                         "SiLU": nn.SiLU,
                         "Sin": Sin,
                         "Siren": Siren,
                         "PReLU": nn.PReLU,
                         "custom": custom,
                         "LeakyReLU": nn.LeakyReLU}
