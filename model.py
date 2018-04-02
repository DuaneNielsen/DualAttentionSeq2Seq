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


class AttentionDecoder(nn.Module):
    def __init__(self, input_dims, sequence_length, cell_size, output_features=1):
        super(AttentionDecoder, self).__init__()
        self.input_dims = input_dims
        self.sequence_length = sequence_length
        self.cell_size = cell_size
        self.lstm = nn.LSTMCell(input_dims, cell_size)
        self.to_output = nn.Linear(cell_size, output_features)

        # attention over encoded timesteps
        self.convolve_encoded_embedding = torch.nn.Conv1d(input_dims, 1, 1)
        self.U = nn.Linear(sequence_length, sequence_length)
        self.W = nn.Linear(cell_size * 2, sequence_length)
        self.V = nn.Linear(sequence_length, sequence_length)
        self.relu_time = nn.ReLU()
        self.softmax_time = nn.Softmax(dim=2)
        self.squash_context_on_input = nn.Linear(self.cell_size + self.input_dims, self.input_dims)
        self.squash_context_on_output = nn.Linear(self.cell_size + self.input_dims, self.cell_size)
        self.output_linear = nn.Linear(self.cell_size, 1)

    def forward(self, input):

        h_t, c_t = self.init_hidden(input.size(0))
        # attention over the encoder states
        e = self.convolve_encoded_embedding(input)

        outputs = []

        for _ in range(self.sequence_length):

            U = self.U(e)
            W = self.W(torch.cat((h_t, c_t), dim=1).unsqueeze(1))
            A = self.relu_time(torch.add(U, W))
            V = self.V(A)
            beta = self.softmax_time(V)

            elements = input * beta
            context = torch.sum(elements, dim=2)
            attentionOnTime = self.squash_context_on_input(torch.cat([context, h_t], dim=1))

            h_t, c_t = self.lstm(attentionOnTime, (h_t, c_t))

            output_context = self.squash_context_on_output(torch.cat([context, h_t], dim=1))
            output = self.output_linear(output_context)
            outputs.append(output)

        return torch.stack(outputs, dim=2)

    def init_hidden(self, batch_size):
        hidden = Variable(next(self.parameters()).data.new(batch_size, self.cell_size), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(batch_size, self.cell_size), requires_grad=False)
        return hidden.zero_(), cell.zero_()