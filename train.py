from data import SetGenerator, PlotOutput
from model import NaiveSeq2Seq
import torch
from torch.autograd import Variable
from sgdr import SGDRScheduler
from tensorboardX import SummaryWriter
import tensorboard_utils as tu
from tqdm import tqdm


def plotResult(model, dataset):
    for minibatch in dataset:
        input = Variable(minibatch[0])
        target = Variable(minibatch[1])
        output = model(input)
        PlotOutput(output, target, title='final result').draw()
        break


def main():

    batch_size = 64
    input_dims = 3
    sequence_length = 10
    cell_size = 2
    encoded_cell_size = 2

    # Learning Rates
    max_rate = 0.1
    min_rate = 0.01
    steps_per_cycle = 3000
    warmup = 30

    # init Tensorboard
    tensorboard_step = 0
    writer = SummaryWriter(comment="DSARNN run " + str(1))

    # grab data
    train, valid = SetGenerator(sequence_length, time_steps=40000, target_index=0)\
        .train_valid(percent_as_float=0.05, batch_size=batch_size)

    # setup model
    model = NaiveSeq2Seq(input_dims, sequence_length, cell_size, encoded_cell_size=2)

    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=max_rate)
    scheduler = SGDRScheduler(optimiser, min_rate, max_rate, steps_per_cycle, warmup, 0)

    for epoch in tqdm(range(70)):

        for minibatch in train:
            input = Variable(minibatch[0])
            target = Variable(minibatch[1])
            optimiser.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimiser.step()
            scheduler.step()
            tensorboard_step += 1
            writer.add_scalar('loss/training loss', loss, tensorboard_step)
            writer.add_scalar('loss/learning rate', tu.get_learning_rate(optimiser), tensorboard_step)

        for minibatch in valid:
            input = Variable(minibatch[0])
            target = Variable(minibatch[1])
            output = model(input)
            loss = criterion(output, target)
            tensorboard_step += 1
            writer.add_scalar('loss/test loss', loss, tensorboard_step)

    plotResult(model, valid)


if __name__ == "__main__":
    main()

