import time
import torch
import numpy as np
import matplotlib.pyplot as plt


class Trainer:

    def __init__(self, model, iterators, criterion, optimizer, device):

        train_iterator, valid_iterator, test_iterator = iterators

        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.test_iterator = test_iterator
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.train_losses = []
        self.valid_losses = []

    def train(self, eps=100, print_every=1, halting_n=4, halting_delta=1e-3, **kwargs):

        ts = []
        train_losses = []
        valid_losses = []
        stored = 0

        for e in range(eps):

            t, train_loss, valid_loss = self.train_evaluate_epoch()
            ts.append(t)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            stored += 1

            if stored == print_every or e + 1 == eps:
                self._print_sequence(ts[e - stored + 1:e + 1],
                                     train_losses[e - stored + 1:e + 1],
                                     valid_losses[e - stored + 1:e + 1],
                                     e - stored + 1, **kwargs)
                stored = 0

            # After computing, test halt condition
            if self.need_halt(valid_losses, halting_n, halting_delta):
                break

        self.train_losses = train_losses
        self.valid_losses = valid_losses

        test_loss = self.evaluate(self.test_iterator)
        self._print_last(ts, train_losses, valid_losses, test_loss, eps, **kwargs)

    @staticmethod
    def need_halt(valid_losses, halting_n, halting_delta):
        if len(valid_losses) >= halting_n:
            return valid_losses[-halting_n] - valid_losses[-1] < halting_delta
        else:
            return False

    def _print_sequence(self, ts, train_losses, valid_losses, epoch, **kwargs):
        # epoch - starting epoch number in sequence
        pass

    def _print_last(self, ts, train_losses, valid_losses, test_loss, epoch, **kwargs):
        # epoch - starting epoch number in sequence
        pass

    def train_log(self, eps, writer, diagram_label, instance_label):
        for e in range(eps):
            _, train_loss, valid_loss = self.train_evaluate_epoch()
            # Write to TensorBoard
            writer.add_scalars(diagram_label, {f"{instance_label} train loss": train_loss,
                                               f"{instance_label} valid loss": valid_loss}, e)

        writer.close()

    def train_evaluate_epoch(self):
        t1 = time.time()
        train_loss = self.train_epoch()
        valid_loss = self.evaluate(self.valid_iterator)
        t2 = time.time()
        t = t2 - t1
        return t, train_loss, valid_loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for i, batch in enumerate(self.train_iterator):
            self.optimizer.zero_grad()
            src, trg, out = self.forward(batch)
            loss = self.criterion(out, trg)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / len(self.train_iterator)
        return mean_loss

    def evaluate(self, iterator):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src, trg, out = self.forward(batch)
                loss = self.criterion(out, trg)
                total_loss += loss.item()

        mean_loss = total_loss / len(iterator)
        return mean_loss

    def forward(self, batch, verbose=False):
        src = batch.src.to(self.device)  # [S, N]
        trg = batch.trg.to(self.device)  # [T + 1, N]

        if verbose:
            print(f'Data received from iterator: src=[{src.shape}]; trg=[{trg.shape}]')

        # Key moment: the -1 index omits the <eos> token
        # This is done because the decoder should never receive <eos> as input
        out = self.model(src, trg[:-1, :])  # [T, N, V]

        if verbose:
            print(f'Data received from model: out=[{out.shape}]')

        # Key moment: we cut off <sos> token from trg, because the model never learns to output it
        # This aligns the out and trg tokens for successful loss calculation
        out = out.reshape(-1, out.shape[2])  # [T * N, V]
        trg = trg[1:].reshape(-1)  # [T * N]

        if verbose:
            print(f'Data reshaped for loss computation: out=[{out.shape}]; trg=[{trg.shape}]')

        return src, trg, out

    def plot_loss(self):
        plt.plot(np.arange(len(self.train_losses)), self.train_losses, color="#e44034", label="Train Loss")
        plt.plot(np.arange(len(self.valid_losses)), self.valid_losses, color="#7d9bda", label="Valid Loss")
        plt.legend()

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")


class VerboseTrainer(Trainer):

    def __init__(self, model, iterators, criterion, optimizer, device):
        super().__init__(model, iterators, criterion, optimizer, device)

    def _print_sequence(self, ts, train_losses, valid_losses, epoch, **kwargs):
        super()._print_sequence(ts, train_losses, valid_losses, epoch, **kwargs)

        sequence_time = sum(ts)
        minutes = int(sequence_time / 60)
        seconds = int(sequence_time % 60)

        if len(ts) > 1:
            epoch_string = f"Epochs {epoch}-{epoch + len(ts) - 1}"
        else:
            epoch_string = f"Epoch {epoch}"

        print(f'\n{epoch_string}: | Time: {minutes}m {seconds}s')
        print(f'Train loss: {train_losses[-1]}')
        print(f'Valid loss: {valid_losses[-1]}')

    def _print_last(self, ts, train_losses, valid_losses, test_loss, epoch, **kwargs):
        super()._print_last(ts, train_losses, valid_losses, test_loss, epoch, **kwargs)

        total_time = sum(ts)
        minutes = int(total_time / 60)
        seconds = int(total_time % 60)

        print(f'\nTotal Time: {minutes}m {seconds}s')
        print(f'Final test loss: {test_loss}')


class TranslatingTrainer(VerboseTrainer):

    def __init__(self, model, iterators, criterion, optimizer, translator, device):
        super().__init__(model, iterators, criterion, optimizer, device)

        self.translator = translator

    def _print_sequence(self, ts, train_losses, valid_losses, epoch, **kwargs):
        super()._print_sequence(ts, train_losses, valid_losses, epoch, **kwargs)

        self.print_translation(**kwargs)

    def _print_last(self, ts, train_losses, valid_losses, test_loss, epoch, **kwargs):
        super()._print_last(ts, train_losses, valid_losses, test_loss, epoch, **kwargs)

        self.print_translation(**kwargs)

    def print_translation(self, translation_phrase="Three brothers are playing football", **kwargs):
        print(self.translator.translate(translation_phrase))
