"""
Returns the current learning rate from the optimizer
"""


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr[0]

