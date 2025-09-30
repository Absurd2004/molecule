# coding=utf-8
import torch
import torch.utils.data as tud
import torch.nn.utils.rnn as tnnur
from importlib import import_module


def _resolve_tqdm():
    try:
        return import_module("tqdm").tqdm
    except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
        def _passthrough(iterable, **kwargs):
            return iterable

        return _passthrough


tqdm = _resolve_tqdm()


class Dataset(tud.Dataset):
    """Dataset that takes a list of SMILES only."""

    def __init__(self, smiles_list, vocabulary, tokenizer):
        """
        Instantiates a Dataset.
        :param smiles_list: A list with SMILES strings.
        :param vocabulary: A Vocabulary object.
        :param tokenizer: A Tokenizer object.
        :return:
        """
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer

        self._encoded_list = []
        for smi in smiles_list:
            enc = self._vocabulary.encode(self._tokenizer.tokenize(smi))
            if enc is not None:
                self._encoded_list.append(enc)

    def __getitem__(self, i):
        return torch.tensor(self._encoded_list[i], dtype=torch.long)  # pylint: disable=E1102

    def __len__(self):
        return len(self._encoded_list)

    @classmethod
    def collate_fn(cls, encoded_seqs):
        return pad_batch(encoded_seqs)


class DecoratorDataset(tud.Dataset):
    """Dataset that takes a list of (scaffold, decoration) pairs."""

    _encoding_cache = {}

    def __init__(self, scaffold_decoration_smi_list, vocabulary):
        self.vocabulary = vocabulary

        cache_key = (id(scaffold_decoration_smi_list), id(self.vocabulary))
        cached_entry = self._encoding_cache.get(cache_key)
        if cached_entry is not None:
            self._encoded_list = cached_entry["encoded"]
            kept, skipped, total = cached_entry["summary"]
            print(f"Reusing cached encoding: kept {kept} pairs, skipped {skipped}, total processed {total}")
            return

        self._encoded_list = []
        skipped_pairs = 0
        if isinstance(scaffold_decoration_smi_list, list):
            pairs = scaffold_decoration_smi_list
        else:
            pairs = list(scaffold_decoration_smi_list)

        for scaffold, dec in tqdm(pairs, desc="Encoding scaffold/decoration pairs", unit="pair"):
            enc_scaff = self.vocabulary.scaffold_vocabulary.encode(
                self.vocabulary.scaffold_tokenizer.tokenize(scaffold))
            enc_dec = self.vocabulary.decoration_vocabulary.encode(
                self.vocabulary.decoration_tokenizer.tokenize(dec))
            if enc_scaff is not None and enc_dec is not None:
                self._encoded_list.append((enc_scaff, enc_dec))
            else:
                skipped_pairs += 1

        total_pairs = len(pairs)
        print(f"Encoding complete: kept {len(self._encoded_list)} pairs, skipped {skipped_pairs}, total processed {total_pairs}")

        self._encoding_cache[cache_key] = {
            "encoded": self._encoded_list,
            "summary": (len(self._encoded_list), skipped_pairs, total_pairs)
        }

    def __getitem__(self, i):
        scaff, dec = self._encoded_list[i]
        return (torch.tensor(scaff, dtype=torch.long), torch.tensor(dec, dtype=torch.long))  # pylint: disable=E1102

    def __len__(self):
        return len(self._encoded_list)

    @classmethod
    def collate_fn(cls, encoded_pairs):
        """
        Turns a list of encoded pairs (scaffold, decoration) of sequences and turns them into two batches.
        :param: A list of pairs of encoded sequences.
        :return: A tuple with two tensors, one for the scaffolds and one for the decorations in the same order as given.
        """
        encoded_scaffolds, encoded_decorations = list(zip(*encoded_pairs))
        return (pad_batch(encoded_scaffolds), pad_batch(encoded_decorations))


def pad_batch(encoded_seqs):
    """
    Pads a batch.
    :param encoded_seqs: A list of encoded sequences.
    :return: A tensor with the sequences correctly padded.
    """
    seq_lengths = torch.tensor([len(seq) for seq in encoded_seqs], dtype=torch.int64)  # pylint: disable=not-callable
    return (tnnur.pad_sequence(encoded_seqs, batch_first=True).cuda(), seq_lengths)
