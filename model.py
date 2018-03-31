import torch
from torch import nn
from torch.autograd import Variable


class SimpleLSTM(nn.Module):
    def __init__(self, input_dims, sequence_length, cell_size, output_features=1):
        super(SimpleLSTM, self).__init__()
        self.input_dims = input_dims
        self.sequence_length = sequence_length
        self.cell_size = cell_size
        self.lstm = nn.LSTMCell(input_dims, cell_size)
        self.to_output = nn.Linear(cell_size, output_features)

    def forward(self, input):

        h_t, c_t = self.init_hidden(input.size(0))

        outputs = []

        for input_t in torch.chunk(input, self.sequence_length, dim=2):
            h_t, c_t = self.lstm(input_t.squeeze(2), (h_t, c_t))
            outputs.append(self.to_output(h_t))

        return torch.stack(outputs, dim=2)

    def init_hidden(self, batch_size):
        hidden = Variable(next(self.parameters()).data.new(batch_size, self.cell_size), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(batch_size, self.cell_size), requires_grad=False)
        return hidden.zero_(), cell.zero_()


class NaiveSeq2Seq(nn.Module):
    def __init__(self, input_dims, sequence_length, cell_size, encoded_cell_size):
        super(NaiveSeq2Seq, self).__init__()
        self.input_dims = input_dims
        self.sequence_length = sequence_length
        self.cell_size = cell_size
        self.encoded_cell_size = encoded_cell_size

        self.encoder = SimpleLSTM(input_dims, sequence_length, cell_size, encoded_cell_size)
        self.decoder = SimpleLSTM(encoded_cell_size, sequence_length, cell_size)

    def forward(self, input):
        encoded = self.encoder(input)
        return self.decoder(encoded)
