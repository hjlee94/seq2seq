from typing import List, Callable
import re
import pickle
import torch

from torch.nn.utils.rnn import pad_sequence

PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
WS = ' '

class Serializable:
    def __init__(self) -> None:
        pass

    @classmethod
    def from_pickle(cls, path):
        with open(path, 'rb') as fd:
            obj = pickle.load(fd)

        return obj

    def dump(self, path):
        with open(path, 'wb') as fd:
            pickle.dump(self, fd)

class BaseTokenizer(Serializable):
    def __init__(self) -> None:
        super().__init__()

        self._tokens = []
        self._token_index = {}

        self.add(PAD)
        self.add(SOS)
        self.add(EOS)
        self.add(WS)

    def get_padding_index(self):
        return self.get(PAD)

    def dump(self, path:str) -> None:
        with open(path, 'wb') as fd:
            pickle.dump(self, fd)

    def add(self, token:str) -> None:
        if token in self._token_index:
            return
        
        idx = self.size()
        self._token_index[token] = idx
        self._tokens.append(token)

    def get(self, token:str) -> int:
        return self._token_index.get(token, self._token_index[WS])
    
    def size(self) -> int:
        return len(self._tokens)
    
    def encode(self, sentence:str) -> List[int]:
        token_list = self.tokenize(sentence)
        return list(map(lambda x:self.get(x), token_list))

    def decode(self, token_list:List[int]) -> str:
        s = []

        for token_id in token_list:
            if token_id == self.get(EOS):
                break
            
            s.append(self._tokens[token_id])

        return ' '.join(s)
    
    def tokenize(self, sentence):
        raise NotImplementedError("tokenize() should be overrided")

class PatternTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self._pattern = re.compile(r'(\'| |\.|,|\t|\W)', re.IGNORECASE)

    def tokenize(self, sentence:str) -> List[str]:
        token_list = self._pattern.split(sentence)
        token_list = list(filter(lambda x: x and x != ' ', token_list))

        for token in token_list:
            self.add(token)

        return token_list

class SimpleTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self._pattern = re.compile(r'(\'| |\.|,|\t|\W)', re.IGNORECASE)

    def tokenize(self, sentence:str) -> List[str]:
        sentence = sentence.lower()
        sentence = re.sub(r'[!\?\.\'\-\"]', ' ', sentence)

        token_list = sentence.split()
        token_list = list(filter(lambda x:x, token_list))

        for token in token_list:
            self.add(token)

        return token_list

class SequenceDataset(Serializable):
    def __init__(self) -> None:
        self._src = []
        self._dst = []

    def add_source(self, x):
        self._src.append(x)

    def add_destination(self, x):
        self._dst.append(x)

    def get_sequence(self, seq, padding_idx):
        padded_seq = pad_sequence(seq, batch_first=True, padding_value=padding_idx)
        return padded_seq

    def get_xy(self, padding_idx=0):
        X = self.get_sequence(self._src, padding_idx)
        y = self.get_sequence(self._dst, padding_idx)

        return X,y

class DatasetHandler:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.dataset = SequenceDataset()

    def parse_txt(self, path:str, filter_func:Callable[[str, str], bool]=None):
        with open(path) as fd:

            for line in fd:
                if not line:
                    continue
                
                line = line.strip()
                src, dst = line.split('\t')

                if filter_func:
                    if filter_func(src, dst):
                        continue

                self.dataset.add_source(
                    torch.tensor(
                        self.tokenizer.encode(src) +
                        [self.tokenizer.get(EOS)]
                        )
                    )
                self.dataset.add_destination(
                    torch.tensor(
                        [self.tokenizer.get(SOS)] + 
                        self.tokenizer.encode(dst) +
                        [self.tokenizer.get(EOS)]
                        )
                    )

if __name__ == '__main__':
    tokenizer = SimpleTokenizer()
    handler = DatasetHandler(tokenizer=tokenizer)
    handler.parse_txt(
        "./dataset/eng-fra.txt", 
        filter_func=lambda src, dst:len(src) > 15
        )

    handler.tokenizer.dump("./rsrc/tokenizer.pkl")
    handler.dataset.dump("./rsrc/dataset.pkl")
    # ds = DatasetHandler.from_pickle("./rsrc/dataset.pkl")

    # ds.dump("./rsrc/dataset.pkl")

    print(handler.tokenizer._token_index)
    print(handler.tokenizer.size())
    X, y = handler.dataset.get_xy()

    print(X)
    print(X.shape)

    print(y)
    print(y.shape)
