import torch.nn as nn
class VAE(nn.Module):
    def __init__(self, input_dims, code_dims, ngpu=4)：
        super(VAE, self).__init__()
        self.name = "abstract_VAE"
        self.input_dims = input_dims
        self.code_dims = code_dims
        self.ngpu = ngpu

    def encode(self, x):
        self.handle_unsupported_op()
        return None

    def decode(self, z):
        self.handle_unsupported_op()
        return None

    def handle_unsupported_op(self):
        print("Unsupported Operation")
        raise(Exception("Unsupported Operation"))    