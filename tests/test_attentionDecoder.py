from unittest import TestCase
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import gradcheck
from model import AttentionDecoder


class TestAttentionDecoder(TestCase):
    def test_forward(self):
        input = (Variable(torch.randn(3, 2, 4).double(), requires_grad=True),)
        test = gradcheck(AttentionDecoder(2, 4, 1).double(), input, eps=1e-6, atol=1e-4)
        print(test)

    def test_forward_1in(self):
        input = (Variable(torch.randn(3, 1, 4).double(), requires_grad=True),)
        test = gradcheck(AttentionDecoder(1, 4, 1).double(), input, eps=1e-6, atol=1e-4)
        print(test)


    def test_forward_dimensions(self):
        input = Variable(torch.randn(3, 2, 4).double(), requires_grad=True)
        output = AttentionDecoder(2, 4, 1).double().forward(input)
        assert len(output.size()) == 3