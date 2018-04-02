from data import SetGenerator, PlotAgreement
from model import AttentionDecoder
import torch
from torch.autograd import Variable
from sgdr import SGDRScheduler
import tensorboard_utils as tu
from tqdm import tqdm
import monitors
from monitors import SummaryWriterWithGlobal


def plotResult(model, dataset, writer):
    for minibatch in dataset:
        input = Variable(minibatch[0]).cuda()
        target = Variable(minibatch[1]).cuda()
        output = model(input)
        chart = PlotAgreement(title='Agreement', output=output, target=target)
        writer.plotImage(chart)
        break


def main():
    for run in range(1):

        batch_size = 64
        input_dims = 1
        sequence_length = 10
        cell_size = 2
        encoded_cell_size = 2

        # Learning Rates
        max_rate = 0.1
        min_rate = 0.03
        steps_per_cycle = 1000
        warmup = 100

        # init Tensorboard
        tensorboard_step = 0
        writer = SummaryWriterWithGlobal(comment="DSARNN run " + str(run))

        # grab data
        train, valid = SetGenerator(sequence_length, time_steps=8000, target_index=0)\
            .train_valid(percent_as_float=0.05, batch_size=batch_size)

        # setup model
        model = AttentionDecoder(input_dims, sequence_length, cell_size).cuda()

        # hooks
        def monitorTemporalAttention(self, input, output):
            if writer.global_step % 10 == 0:
                monitors.monitorSoftmax(self, input, output, 'temporal', writer, dim=2)
        model.softmax_time.register_forward_hook(monitorTemporalAttention)

        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.SGD(model.parameters(), lr=max_rate)
        scheduler = SGDRScheduler(optimiser, min_rate, max_rate, steps_per_cycle, warmup, 0)

        for epoch in tqdm(range(1800)):

            for minibatch in train:
                input = Variable(minibatch[0]).cuda()
                target = Variable(minibatch[1]).cuda()
                optimiser.zero_grad()
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimiser.step()
                scheduler.step()
                writer.step()
                writer.add_scalar('loss/training loss', loss, writer.global_step)
                writer.add_scalar('loss/learning rate', tu.get_learning_rate(optimiser), writer.global_step)

            for minibatch in valid:
                input = Variable(minibatch[0]).cuda()
                target = Variable(minibatch[1]).cuda()
                output = model(input)
                loss = criterion(output, target)
                writer.step()
                writer.add_scalar('loss/test loss', loss, writer.global_step)

            if epoch % 20 == 0:
                plotResult(model, valid, writer)


if __name__ == "__main__":
    main()

