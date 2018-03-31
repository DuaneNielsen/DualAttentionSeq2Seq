from unittest import TestCase
from data import SetGenerator, PlotOutput


class TestPlotOutput(TestCase):
    def test_draw(self):
        train, valid = SetGenerator(10, 400, 0).train_valid(0.1, batch_size=1)

        target = train.__iter__().__next__()[1]

        PlotOutput(target, target, filename='test', title='test_target.html').draw()
