from data import SetGenerator, PlotAgreement
from model import DualAttentionSeq2Seq
import torch
from torch.autograd import Variable
from sgdr import SGDRScheduler
from tqdm import tqdm
import monitors
from monitors import SummaryWriterWithGlobal


def plotResult(model, dataset, writer):
    for minibatch in dataset:
        input = Variable(minibatch[0])
        target = Variable(minibatch[1])
        output = model(input)
        chart = PlotAgreement(title='Agreement', output=output, target=target)
        writer.plotImage(chart)
        break


def main():
    for run in range(1):

        batch_size = 64
        input_dims = 3
        sequence_length = 10
        cell_size = 2
        encoded_cell_size = 1

        # Learning Rates
        max_rate = 0.18
        min_rate = 0.04
        steps_per_cycle = 20000
        warmup = 100

        # init Tensorboard
        writer = SummaryWriterWithGlobal(comment="DSARNN run " + str(run))

        # grab data
        train, valid = SetGenerator(sequence_length, time_steps=8000, target_index=0)\
            .train_valid(percent_as_float=0.05, batch_size=batch_size)

        # setup model
        model = DualAttentionSeq2Seq(input_dims, sequence_length, cell_size, encoded_cell_size)
        model.registerHooks(writer)

        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.SGD(model.parameters(), lr=max_rate)
        scheduler = SGDRScheduler(optimiser, min_rate, max_rate, steps_per_cycle, warmup, 0)

        # around 20 - 40 k epochs for training
        for epoch in tqdm(range(1)):

            for minibatch in train:
                input = Variable(minibatch[0])
                target = Variable(minibatch[1])
                optimiser.zero_grad()
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimiser.step()
                scheduler.step()
                writer.step()
                writer.add_scalar('loss/training loss', loss, writer.global_step)
                writer.add_scalar('loss/learning rate', monitors.get_learning_rate(optimiser), writer.global_step)

            for minibatch in valid:
                input = Variable(minibatch[0])
                target = Variable(minibatch[1])
                output = model(input)
                loss = criterion(output, target)
                writer.step()
                writer.add_scalar('loss/test loss', loss, writer.global_step)

            if epoch % 200 == 0:
                plotResult(model, valid, writer)


if __name__ == "__main__":
    main()

