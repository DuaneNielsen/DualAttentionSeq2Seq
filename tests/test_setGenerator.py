from unittest import TestCase
from data import SetGenerator
import math
from torch.utils.data import DataLoader


class TestSetGenerator(TestCase):
    def test_generateSet(self):
        sequence_length = 10
        time_steps = 4000
        target_index = 0

        dataset = SetGenerator(sequence_length, time_steps, target_index).generateSet()
        assert len(dataset) == (time_steps / sequence_length) - 1

    def test_train_valid(self):
        sequence_length = 10
        time_steps = 4000
        target_index = 0

        dataset = SetGenerator(sequence_length, time_steps, target_index).generateSet()
        train, valid = SetGenerator(sequence_length, time_steps, target_index).train_valid(0.1, batch_size=1)
        length_valid = len(valid)
        length_train = len(train)
        length_total = len(dataset)
        assert len(valid) == math.floor(len(dataset) * 0.1)



