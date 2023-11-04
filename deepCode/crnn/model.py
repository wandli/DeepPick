import torch
import torch.nn as nn
from torch.nn import functional as F


class CRNN(nn.Module):
    def __init__(self, batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab,
                 vocab_size, embedding_length, weights):
        super(CRNN, self).__init__()
        """
        :param batch_size:
        :param output_size:
        :param in_channels:
        :param out_channels:
        :param kernel_heights: 3 different kernel_heights
        :param stride:
        :param padding:
        :param keep_probab: dropout probability
        :param vocab_size: Size of the vocabulary containing unique words
        :param embedding_length:
        :param weights: Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        """
        self.feature = ''
        self.batch_size = batch_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.keep_probab = keep_probab
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.weights = weights

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight = nn.Parameter(torch.Tensor(weights), requires_grad=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
        self.lstm = nn.LSTM(input_size=3, hidden_size=15, num_layers=2, bidirectional=True)
        self.dropout = nn.Dropout(keep_probab)
        self.label = nn.Linear(len(kernel_heights) * out_channels, output_size)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(
            2)  # max_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, input_sentences):
        inputs = self.word_embeddings(input_sentences)
        # input.size() = (batch_size, num_seq, embedding_length)
        inputs = inputs.unsqueeze(0)    # batch_size
        inputs = inputs.unsqueeze(1)    # channel
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        max_out1 = self.conv_block(inputs, self.conv1)
        max_out2 = self.conv_block(inputs, self.conv2)
        max_out3 = self.conv_block(inputs, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        lstm_input_split = all_out.split(3, 1)
        lstm_input = torch.cat((lstm_input_split[0], lstm_input_split[1]), 0)
        for i in range(2, 10):
            lstm_input = torch.cat((lstm_input, lstm_input_split[i]), 0)
        lstm_input = lstm_input.unsqueeze(1)
        lstm_output, (hn, cn) = self.lstm(lstm_input)
        lstm_output = lstm_output[-1]
        self.feature = lstm_output
        fc_in = self.dropout(lstm_output)
        logits = self.label(fc_in)
        return logits

