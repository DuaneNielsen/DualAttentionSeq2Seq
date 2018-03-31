from unittest import TestCase
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import gradcheck
from model import SimpleLSTM


class TestSimpleLSTM(TestCase):
    def test_gradcheck_sanity(self):
        input = (Variable(torch.randn(20, 20).double(), requires_grad=True),)
        test = gradcheck(nn.Linear(20, 1).double(), input, eps=1e-6, atol=1e-4)
        print(test)

    def test_forward(self):
        input = ( Variable(torch.randn(1, 2, 4).double(), requires_grad=True),)
        test = gradcheck(SimpleLSTM(2, 4, 1).double(), input, eps=1e-6, atol=1e-4)
        print(test)

