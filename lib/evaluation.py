import time
import torch
from torchtext.data.metrics import bleu_score


class Translator:
    def __init__(self, model, spacy_model, field_src, field_trg, device):
        self.model = model
        self.spacy_model = spacy_model
        self.field_src = field_src
        self.field_trg = field_trg
        self.device = device

    def translate(self, sentence, verbose=False):
        if isinstance(sentence, str):
            tokens = [token.text.lower() for token in self.spacy_model(sentence)]
        else:
            tokens = [token.lower() for token in sentence]
        tokens = [self.field_src.init_token] + tokens + [self.field_src.eos_token]
        translation = self.translate_tokens(tokens, verbose)
        return translation

    def translate_tokens(self, tokens, verbose=False):
        self.model.eval()
        idx = [self.field_src.vocab.stoi[token] for token in tokens]
        tensor = torch.LongTensor(idx).unsqueeze(1).to(self.device)

        if verbose:
            print(f'Tokenized data ready for manual translation: tensor=[{tensor.shape}]')

        sos = self.field_trg.vocab.stoi["<sos>"]
        eos = self.field_trg.vocab.stoi["<eos>"]
        target = [sos]

        for i in range(20):
            trg_tensor = torch.LongTensor(target).unsqueeze(1).to(self.device)

            with torch.no_grad():
                out = self.model(tensor, trg_tensor)

            if verbose:
                print(f'Time step {i}: tensor=[{tensor.shape}]; trg_tensor=[{trg_tensor.shape}]; out=[{out.shape}]')

            choice = out.argmax(2)[-1, :].item()
            target.append(choice)

            if choice == eos:
                break

        translation = [self.field_trg.vocab.itos[i] for i in target]

        if verbose:
            print(f'The final result has {len(translation) - 1} tokens (<sos> excluded)')

        return translation[1:]

    def calculate_bleu(self, data, verbose=False):
        t1 = time.time()
        trgs = []
        pred_trgs = []

        for datum in data:
            src = vars(datum)['src']
            trg = vars(datum)['trg']

            pred_trg = self.translate(src)[:-1]

            pred_trgs.append(pred_trg)
            trgs.append([trg])

        score = bleu_score(pred_trgs, trgs)

        t2 = time.time()
        minutes = int((t2 - t1) / 60)
        seconds = int((t2 - t1) % 60)

        if verbose:
            print(f'\nTotal Time: {minutes}m {seconds}s')

        return score * 100
