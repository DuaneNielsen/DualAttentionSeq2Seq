from unittest import TestCase
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import gradcheck
from model import AttentionEncoder


class TestAttentionEncoder(TestCase):
    def test_forward(self):
        input = (Variable(torch.randn(1, 2, 4).double(), requires_grad=True),)
        test = gradcheck(AttentionEncoder(2, 4, 1).double(), input, eps=1e-6, atol=1e-4)
        print(test)

