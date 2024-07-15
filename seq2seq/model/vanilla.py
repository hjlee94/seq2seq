import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()

        self._embedding_size = 64
        self._hidden_size = 128
        self._padding_idx = padding_idx

        self._encoder  = Encoder(
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            num_layer=2,
            embedding_size=self._embedding_size, 
            hidden_size=self._hidden_size)
        
        self._decoder = Decoder(
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            num_layer=2,
            embedding_size=self._embedding_size,
            hidden_size=self._hidden_size
        )

    def forward(self, src, lengths, dst, teacher_forcing_ratio=0.5):
        context = self._encoder(src, lengths)
        res = self._decoder(dst, context, teacher_forcing_ratio)

        return res
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, padding_idx, num_layer=1, embedding_size=32, hidden_size=256):
        super().__init__()
        self._embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size, 
            padding_idx=padding_idx)
        
        self._rnn = nn.LSTM(
            input_size=embedding_size, 
            hidden_size=hidden_size, 
            num_layers=num_layer,
            batch_first=True,
            dropout=0.1,
            )

    def forward(self, src, lengths):
        X = self._embedding(src)
        X = pack_padded_sequence(X, lengths, batch_first=True)
        _, hs = self._rnn(X)

        return hs
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, padding_idx, num_layer=1, embedding_size=32, hidden_size=256) -> None:
        super().__init__()
        self._vocab_size = vocab_size

        self._embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size, 
            padding_idx=padding_idx)

        self._rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
            dropout=0.1,
        )

        self._dropout = nn.Dropout(p=0.1)

        self._fc1 = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size)
        
    def forward(self, dst, context, teacher_forcing_ratio=0.5):
        target_length = dst.shape[1] - 1

        inp = dst[:, 0].unsqueeze(1)

        res = []

        for i in range(target_length):
            out = self._embedding(inp) # (N, 1) -> (N, 1, 32)

            out, hs = self._rnn(out, context) # (N, 1, D*H=256)
            out = out.squeeze(1) # (N, 256)

            out = self._dropout(out)

            out = self._fc1(out) # (N, 256) -> (N, VOCAB_SIZE)

            res.append(out.unsqueeze(1))

            if torch.rand(1).item() < teacher_forcing_ratio:
                inp = dst[:, i+1].unsqueeze(1)
            else:
                inp = out.argmax(dim=1).unsqueeze(1)
        
        res = torch.cat(res, dim=1)

        return res