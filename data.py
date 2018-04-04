import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.utils.data
import torch.utils.data.sampler
import math
import monitors

class ChunkSampler(torch.utils.data.sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, start, num_samples, ):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


"""
(sequence_length, minibatch_size, input_dims)
"""


class SetGenerator:

    def __init__(self, sequence_length, time_steps, target_index):
        self.sequence_length = sequence_length
        self.time_steps = time_steps
        self.target_index = target_index

    def generateSet(self):

        line = np.linspace(0, self.time_steps, self.time_steps)

        rows = []

        x = np.mod(line, 30)
        x[x > 20] = 20
        x = x.reshape(self.time_steps, 1)

        rows.append(np.sin(line / 1.5).reshape(self.time_steps, 1))
        rows.append(x)
        rows.append(np.random.randn(self.time_steps).reshape(self.time_steps, 1))

        data = np.concatenate(rows, axis=1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)

        # data is (time_step, input) we need to split it into time windows of batch ( batch, input, timestep )

        # convert to (input, time_step)
        raw_tensor = torch.FloatTensor(data).transpose(0, 1)

        # split into batch rows of sequence length each)
        batch_rows = torch.split(raw_tensor, self.sequence_length, dim=1)

        # the last batch row is an offcut, as it might be the wrong size, throw it away
        batch_rows = batch_rows[:-1]

        # now we have (batch, input, time_step in range(sequence_length))
        final_tensor = torch.stack(batch_rows, dim=0)

        # compuate target
        target_tensor = final_tensor[:, self.target_index, :].clone().unsqueeze(1)

        # wrap and return
        return torch.utils.data.TensorDataset(final_tensor, target_tensor)


    """
    provide a percent in range 0.0 .. 1.0
    """
    def train_valid(self, percent_as_float, batch_size):
        dataset = self.generateSet()
        val_size = math.floor(len(dataset) * percent_as_float)
        train_size = math.floor(len(dataset) * (1.0 - percent_as_float))

        loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(0, train_size))
        loader_val = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(train_size, val_size))

        return loader_train, loader_val


"""
accepts vector of points, plots first input and first element of batch

"""


class PlotAgreement(monitors.LinePlot):
    def __init__(self, title, output, target):
        super(PlotAgreement, self).__init__(title)
        self.addLine(output[0, 0, :])
        self.addLine(target[0, 0, :])
