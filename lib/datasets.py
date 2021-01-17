import spacy
import torch
import torchtext
from torchtext.data import Field, BucketIterator


def get_multi30k(bs, src_lang="en", trg_lang="de"):

    field_src = Field(tokenize="spacy",
                      init_token='<sos>',
                      eos_token='<eos>',
                      tokenizer_language=src_lang,
                      lower=True)

    field_trg = Field(tokenize="spacy",
                      init_token='<sos>',
                      eos_token='<eos>',
                      tokenizer_language=trg_lang,
                      lower=True)

    data = torchtext.datasets.Multi30k.splits((f'.{src_lang}', f'.{trg_lang}'),
                                              [field_src, field_trg])

    field_src.build_vocab(data[0], min_freq=2)
    field_trg.build_vocab(data[0], min_freq=2)

    src_vocab = len(field_src.vocab)
    trg_vocab = len(field_trg.vocab)

    src_pad_idx = field_src.vocab.stoi['<pad>']
    trg_pad_idx = field_trg.vocab.stoi['<pad>']

    sp_src = spacy.load(src_lang)
    sp_trg = spacy.load(trg_lang)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    iterators = BucketIterator.splits(data, batch_size=bs, device=device, shuffle=True)

    return field_src, field_trg, src_vocab, trg_vocab, src_pad_idx, trg_pad_idx, sp_src, sp_trg, device, data, iterators
