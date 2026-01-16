import torch.quantization as tq
import torch.nn as nn

class QATWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
