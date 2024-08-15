import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch.nn as nn
import torch.functional as F
from sklearn.preprocessing import StandardaccakldlkmScaler
from sklearn.model_selection import train_test_split

# for the transformer architecture we need some functions


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        # assert returns falsse if the condition is not satisfied
        assert d_model % num_heads == 0, "d_model should be divisible by the number of heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product(self, Q, K, V, mask=None):
        attention_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        # view is like the reshpae in numpy
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_head(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_k(K))

        attention_output = self.scaled_dot_product(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attention_output))

        return output

## Position wise Feed Forward Network ##


class PositonWiseForward(nn.Module):
    def __init__(self, d_model, d_diff):
        super(PositonWiseForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_diff)
        self.fc2 = nn.Linear(d_diff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class Transformer:

    def foo():
        return


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.input = input_dim
        self.output_dim = output_dim
        self.activation = nn.sigmoid()

        return


class Decoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        return
