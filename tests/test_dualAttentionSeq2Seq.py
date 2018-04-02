from unittest import TestCase
from model import DualAttentionSeq2Seq
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import gradcheck


class TestDualAttentionSeq2Seq(TestCase):
    def test_forward(self):
        batch_size = 13
        input_dims = 5
        seq_len = 7
        cell_size = 3
        encoded_cell = 2
        input = (Variable(torch.randn(batch_size, input_dims, seq_len).double(), requires_grad=True),)
        test = gradcheck(DualAttentionSeq2Seq(input_dims, seq_len, cell_size, encoded_cell_size=encoded_cell).double(),
                         input, eps=1e-6, atol=1e-4)
        print(test)
