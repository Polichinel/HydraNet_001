
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):

    """
    A basic ConvLSTM cell. The input and output shapes are the same. Give the network a way to regulate how much of the previous state is kept and how much is overwritten.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Input gate
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=True) # if it runs, try to remove bias - you are using batchnorm after all
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=True)

    def forward(self, x, hidden):
        h, c = hidden

        # Input gate
        i_t = torch.sigmoid(self.Wxi(x) + self.Whi(h))      # i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi})
        # Forget gate
        f_t = torch.sigmoid(self.Wxf(x) + self.Whf(h))      # f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf})
        # Cell state
        c_tilde = torch.tanh(self.Wxc(x) + self.Whc(h))     # g = tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg})   (where g = c_tilde)
        c = f_t * c + i_t * c_tilde                         # c' = f * c + i * g
        # Output gate
        o_t = torch.sigmoid(self.Wxo(x) + self.Who(h))      # o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho})
        h = o_t * torch.tanh(c)                             # h' = o * \tanh(c') \\

        return h, c

# Example usage
# input_channels = 3
# hidden_channels = 10
# kernel_size = 3
# cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
# x = torch.randn(1, input_channels, 32, 32)
# hidden_state = (torch.randn(1, hidden_channels, 32, 32), torch.randn(1, hidden_channels, 32, 32))

# output, new_hidden_state = cell(x, hidden_state)
