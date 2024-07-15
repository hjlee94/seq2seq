import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttentionSeq2Seq(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()

        self._embedding_size = 64
        self._hidden_size = 128

        self._encoder  = AttentionEncoder(
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            num_layer=2,
            embedding_size=self._embedding_size, 
            hidden_size=self._hidden_size)
        
        self._decoder = AttentionDecoder(
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            num_layer=2,
            embedding_size=self._embedding_size,
            hidden_size=self._hidden_size
        )

    def forward(self, src, lengths, dst, teacher_forcing_ratio=0.5):
        enc_out, enc_hs = self._encoder(src, lengths)
        res = self._decoder(dst, enc_out, enc_hs, teacher_forcing_ratio)

        return res

class AttentionEncoder(nn.Module):
    def __init__(self, vocab_size, padding_idx, num_layer=2, embedding_size=32, hidden_size=256):
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
            dropout=0.2)

    def forward(self, src, lengths):
        X = self._embedding(src)
        X = pack_padded_sequence(X, lengths, batch_first=True)
        out, hs = self._rnn(X) 

        return out, hs
    
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx, num_layer=2, embedding_size=32, hidden_size=256) -> None:
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
            dropout=0.2,
        )

        self._attention = Attention()

        self._dropout = nn.Dropout(p=0.1)

        self._fc1 = nn.Linear(
            in_features=hidden_size * 2,
            out_features=vocab_size)
        
    def forward(self, dst, enc_out, enc_hs, teacher_forcing_ratio=0.5):
        target_length = dst.shape[1] - 1
        inp = dst[:, 0].unsqueeze(1)
        res = []

        for i in range(target_length):
            out = self._embedding(inp) # (N, 1) -> (N, 1, 32)

            out, _ = self._rnn(out, enc_hs) # (N, 1, D*H=256)

            context_vector = self._attention(out, enc_out)

            out = torch.cat((context_vector, out.squeeze(1)), dim=1)

            out = self._dropout(out)

            out = self._fc1(out) # (N, 256) -> (N, VOCAB_SIZE)
            res.append(out.unsqueeze(1))

            if torch.rand(1).item() < teacher_forcing_ratio:
                inp = dst[:, i+1].unsqueeze(1)
            else:
                inp = out.argmax(dim=1).unsqueeze(1)

        res = torch.cat(res, dim=1)

        return res

class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self._softmax = nn.Softmax(dim=1)

    def forward(self, dec_out, enc_out):
        enc_out, _ = pad_packed_sequence(enc_out, batch_first=True)

        attention_weight = torch.bmm(dec_out, enc_out.transpose(1,2))
        attention_weight = attention_weight.squeeze(1)
        attention_weight = self._softmax(attention_weight)
        
        # ==== weighted sum : element-wise ====            
        # attention_weight = attention_weight.unsqueeze(2)
        # context_vector = torch.sum(enc_out * attention_weight, dim=1)
        
        # ==== weighted sum : bmm ====
        # # weight (N, 1, L) @ enc_out (N, L, 256) -> (N, 1, 256)
        out = torch.bmm(attention_weight.unsqueeze(1), enc_out)
        out = out.squeeze(1)

        return out

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

        self._softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = self._softmax(scores)
        context = torch.bmm(weights, keys)

        return context, weights