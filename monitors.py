import math
from tensorboardX import SummaryWriter


"""
Compuates the entropy of a softmax
(batch, output_value)
"""


class SoftMaxEntropy:

    @staticmethod
    def entropy(softmax_output):
        sfo = softmax_output.data
        entropy = (- sfo * sfo.log()).sum(1)
        return entropy

    @staticmethod
    def aveEntropy(softmax_output):
        entropy = SoftMaxEntropy.entropy(softmax_output)
        return entropy.sum() / softmax_output.size(0)

    @staticmethod
    def maxEntropy(bins):
        even_odds = 1/bins
        max_entropy = (- even_odds * math.log(even_odds)) * bins
        return max_entropy


"""
Forward Hook for monitoring attention state and recording it to tensoboard
"""


def monitorSoftmax(self, input, output, name, writer, tensorboard_step, granular=False):

    entropy = SoftMaxEntropy.aveEntropy(output)
    max_entropy = SoftMaxEntropy.maxEntropy(output.size(1))
    writer.add_scalar('entropy/' + name + 'max: ' + max_entropy, entropy, tensorboard_step)

    if granular:
        # input is a tuple of packed inputs
        # output is a Variable. output.data is the Tensor we are interested
        for i in range(output.data.size()[1]):
            writer.add_scalar(name + '/attention_' + str(i), output.data[0, i], tensorboard_step)


class SummaryWriterWithGlobal(SummaryWriter):
    def __init__(self, comment):
        super(comment=comment)
        self.step = 0

    def step(self):
        self.step += 1


