import math
from tensorboardX import SummaryWriter


"""
Compuates the entropy of a softmax
(batch, output_value)
"""


class SoftMaxEntropy:

    @staticmethod
    def entropy(softmax_output, dim):
        sfo = softmax_output.data
        entropy = (- sfo * sfo.log()).sum(dim)
        return entropy

    @staticmethod
    def aveEntropy(softmax_output, dim):
        entropy = SoftMaxEntropy.entropy(softmax_output, dim)
        return entropy.sum() / softmax_output.size(0)

    @staticmethod
    def maxEntropy(bins):
        even_odds = 1.0/bins
        max_entropy = (- even_odds * math.log(even_odds)) * bins
        return max_entropy


"""
Forward Hook for monitoring attention state and recording it to tensoboard
"""
#todo make granular mode accept arbitrary dims

def monitorSoftmax(self, input, output, name, writer, granular=False, dim=1):

    entropy = SoftMaxEntropy.aveEntropy(output, dim)
    max_entropy = SoftMaxEntropy.maxEntropy(output.size(dim))
    writer.add_scalar('entropy/' + name + 'max: ' + str(max_entropy), entropy, writer.global_step)

    if granular:
        # input is a tuple of packed inputs
        # output is a Variable. output.data is the Tensor we are interested
        for i in range(output.data.size()[dim]):
            writer.add_scalar(name + '/attention_' + str(i), output.data[0, i], writer.global_step)


class SummaryWriterWithGlobal(SummaryWriter):
    def __init__(self, comment):
        super(SummaryWriterWithGlobal, self).__init__()
        self.global_step = 0

    def step(self):
        self.global_step += 1


