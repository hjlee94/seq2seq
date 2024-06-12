from data import DatasetHandler, SimpleTokenizer, SequenceDataset
from model import Seq2Seq, PeekySeq2Seq, AttentionSeq2Seq

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torch
import time
import os

class TorchDataset(Dataset):
    def __init__(self, X, y):
        self._X = X
        self._y = y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        return self._X[idx], self._y[idx]

class Trainer:
    def __init__(self, padding_idx) -> None:
        self._model = None

        self.epoch = 10
        self.lr = 1e-3
        self.padding_idx = padding_idx

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.history = []

    def init_history(self):
        self.history = []

    def set_model(self, model):
        self._model = model

    def load_pretrained_params(self, path):
        self._model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def checkpoint(self, out_path):
        torch.save(self._model.state_dict(), out_path)

    def _train(self, train_dl, test_dl):
        if torch.cuda.is_available():
            self._model = self._model.cuda()

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.padding_idx)

        for e in range(self.epoch):
            self._model.train()

            epoch_loss = 0
            s0 = time.time()

            for i, batch in enumerate(train_dl):
                X_batch, y_batch = batch

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                lengths = torch.sum(X_batch != self.padding_idx, dim=1).cpu()
                idx = torch.argsort(lengths, descending=True)

                X_batch = X_batch[idx]; lengths = lengths[idx]
                y_batch = y_batch[idx]

                optimizer.zero_grad()

                result = self._model(X_batch, lengths, y_batch, 0.5).to(self.device)

                n_out = result.shape[-1]

                result_flatten = result.reshape(-1, n_out) # (N * L, vocab_prob)
                y_batch_flatten = y_batch[:, 1:].reshape(-1) # (N * L,)

                loss = criterion(result_flatten, y_batch_flatten)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(train_dl)

            eval_loss = self._evaluate(test_dl)

            e0 = time.time()
            elapsed_time = e0 - s0 

            print(f"[epoch: {e:02d}] loss:{epoch_loss:.4f} eval:{eval_loss:.4f} time:{elapsed_time:.3f}s")

            self.history.append({
                'loss':loss,
                'eval':eval_loss,
                'time': elapsed_time
            })

    def _evaluate(self, test_dl):
        self._model.eval()

        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        epoch_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(test_dl):
                X_batch, y_batch = batch

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                lengths = torch.sum(X_batch != self.padding_idx, dim=1).cpu()
                idx = torch.argsort(lengths, descending=True)

                X_batch = X_batch[idx]; lengths = lengths[idx]
                y_batch = y_batch[idx]

                result = self._model(X_batch, lengths, y_batch, 0.0).to(self.device)

                n_out = result.shape[-1]

                result_flatten = result.reshape(-1, n_out) # (N * L, vocab_prob)
                y_batch_flatten = y_batch[:, 1:].reshape(-1) # (N * L,)

                loss = criterion(result_flatten, y_batch_flatten)

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(test_dl)

        return epoch_loss

    def train_model(self, X, y):
        self.init_history()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        train_dataset = TorchDataset(X_train, y_train)
        train_dl = DataLoader(
            dataset=train_dataset, 
            batch_size=512, 
            shuffle=True
            )
        
        test_dataset = TorchDataset(X_test, y_test)
        test_dl = DataLoader(
            dataset=test_dataset, 
            batch_size=512, 
            shuffle=True
            )
        
        self._train(train_dl, test_dl)

if __name__ == '__main__':

    ds = SequenceDataset.from_pickle("./rsrc/dataset.pkl")
    tokenizer = SimpleTokenizer.from_pickle("./rsrc/tokenizer.pkl")
    X, y = ds.get_xy()

    model = AttentionSeq2Seq(
        vocab_size=tokenizer.size(),
        padding_idx=tokenizer.get_padding_index()
        )

    # model = PeekySeq2Seq(
    #     vocab_size=tokenizer.size(),
    #     padding_idx=tokenizer.get_padding_index()
    # )
    
    # model = Seq2Seq(
    #     vocab_size=tokenizer.size(),
    #     padding_idx=tokenizer.get_padding_index()
    #     )

    trainer = Trainer(
        padding_idx=tokenizer.get_padding_index()
    )

    trainer.set_model(model)

    # pretrained_path = './rsrc/seq2seq-vanila.pt'
    pretrained_path = ''
    if pretrained_path:
        trainer.load_pretrained_params(pretrained_path)

    trainer.epoch = 100
    trainer.lr = 1e-3

    trainer.train_model(X, y)
    trainer.checkpoint("./rsrc/seq2seq-attention.pt")
    