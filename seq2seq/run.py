from data import DatasetHandler, SimpleTokenizer, SOS, EOS, SequenceDataset
from model import Seq2Seq, PeekySeq2Seq, AttentionSeq2Seq
# from model.test import Seq2Seq

import torch

import numpy as np


def preprocess(msg, tokenizer):
    token_list = tokenizer.encode(msg) + [tokenizer.get(EOS)]
    x = np.array(token_list).reshape(1,-1)
    return x


def run(token_path, model_path):
    tokenizer = SimpleTokenizer.from_pickle(token_path)

    model = AttentionSeq2Seq(
        vocab_size=tokenizer.size(),
        padding_idx=tokenizer.get_padding_index()
        )
    # model = PeekySeq2Seq(
    #     vocab_size=tokenizer.size(),
    #     padding_idx=tokenizer.get_padding_index()
    #     )
    # model = Seq2Seq(
    #     vocab_size=tokenizer.size(),
    #     padding_idx=tokenizer.get_padding_index()
    #     )
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()
    prompt = input('>> ')

    while prompt and prompt != "q":
        print(f"got : {prompt}")
        x = preprocess(prompt, tokenizer)

        print(f"tokens : {x}")

        with torch.no_grad():
            x = torch.from_numpy(x).long()
            length = torch.tensor([x.shape[1]])
            y = torch.zeros((1, 15)).long()
            y = y.fill_(tokenizer.get(SOS))

            res = model(x, length, y, 0.0)
            res = torch.argmax(res, dim=2)

            answer = tokenizer.decode(res[0])
            print(f'answer: {answer}')

        prompt = input('>> ')


if __name__ == '__main__':
    run(token_path="./rsrc/tokenizer.pkl", model_path="./rsrc/seq2seq-attention.pt")

    
