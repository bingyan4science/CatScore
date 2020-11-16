import linecache
import os
import subprocess
from collections import Counter
from typing import List, Optional

import torch
import torchtext
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        
        self._file_path = file_path
        self._len = int(subprocess.check_output("wc -l " + file_path, shell=True).split()[0]) - 1
        self.tokenizer = tokenizer
        self.max_length = block_size
        
    def __getitem__(self, idx):
        line = linecache.getline(self._file_path, idx + 1).strip()
        sample = self.tokenizer(
                        line,
                        max_length=self.max_length,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        return sample['input_ids'].squeeze()
      
    def __len__(self):
        return self._len


class MolTranslationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        type_path="train",
        max_source_length=500,
        max_target_length=500,
    ):
        super().__init__()
        # FIXME: the rstrip logic strips all the chars, it seems.
        # tok_name = tokenizer.__class__.__name__.lower().rstrip("tokenizer")

        self._source_path = os.path.join(data_dir, type_path + ".source")
        self._target_path = os.path.join(data_dir, type_path + ".target")
        self._len_source = int(subprocess.check_output("wc -l " + self._source_path, shell=True).split()[0]) - 1
        self._len_target = int(subprocess.check_output("wc -l " + self._target_path, shell=True).split()[0]) - 1
        assert self._len_source == self._len_target, "Source file and target file don't match!"
        self.tokenizer = tokenizer
        self.max_source_len = max_source_length
        self.max_target_len = max_target_length

        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return self._len_source

    def __getitem__(self, idx):
        source_line = linecache.getline(self._source_path, idx + 1).strip()
        source_sample = self.tokenizer(
                        source_line,
                        max_length=self.max_source_len,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        target_line = linecache.getline(self._target_path, idx + 1).strip()
        target_sample = self.tokenizer(
                        target_line,
                        max_length=self.max_target_len,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        source_ids = source_sample["input_ids"].squeeze()
        target_ids = target_sample["input_ids"].squeeze()
        src_mask = source_sample["attention_mask"].squeeze()
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "decoder_input_ids": target_ids}

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


class MolTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a Molecular tokenizer. Based on SMILES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`, `optional`, defaults to ''):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<blank>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """

    def __init__(
        self,
        vocab_file='',
        source_files='',
        unk_token='<unk>',
        bos_token='<s>',
        pad_token="<blank>",
        eos_token='</s>',
        vocab_size=None,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            **kwargs)

        self.create_vocab(vocab_file=vocab_file, source_files=source_files, vocab_size=vocab_size)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def merge_vocabs(self, vocabs, vocab_size=None):
        """
        Merge individual vocabularies (assumed to be generated from disjoint
        documents) into a larger vocabulary.
        Args:
            vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
            vocab_size: `int` the final vocabulary size. `None` for no limit.
        Return:
            `torchtext.vocab.Vocab`
        """
        merged = sum([vocab.freqs for vocab in vocabs], Counter())
        return torchtext.vocab.Vocab(merged,
                                     specials=list(self.special_tokens_map.values()),
                                     max_size=vocab_size)

    def create_vocab(self, vocab_file=None, source_files=None, vocab_size=None):
        """
        Create a vocabulary from current vocabulary file or from source file(s).
        Args:
            vocab_file (:obj:`string`, `optional`, defaults to ''):
                File containing the vocabulary (torchtext.vocab.Vocab class).
            source_files (:obj:`string`, `optional`, defaults to ''):
                File containing source data files, vocabulary would be built based on the source file(s).
        """
        if (not vocab_file) and (not source_files):
            self.vocab = []
        if vocab_file:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    "Can't find a vocabulary file at path '{}'.".format(vocab_file)
                )
            else:
                self.vocab = torch.load(vocab_file)

        if source_files:
            if isinstance(source_files, str):
                if not os.path.isfile(source_files):
                    raise ValueError(
                        "Can't find a source file at path '{}'.".format(source_files)
                    )
                else:
                    source_files = [source_files]
            counter = {}
            vocabs = {}
            for i, source_file in enumerate(source_files):
                counter[i] = Counter()
                with open(source_file) as rf:
                    for line in tqdm(rf, desc='Generating {}'.format(source_file)):
                        try:
                            items = self._tokenize(line.strip())
                            counter[i].update(items)
                        except AssertionError:
                            print(line.strip())
                specials = list(self.special_tokens_map.values())
                vocabs[i] = torchtext.vocab.Vocab(counter[i], specials=specials)
            self.vocab = self.merge_vocabs([vocabs[i] for i in range(len(source_files))], vocab_size=vocab_size)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        """
        Tokenize a SMILES molecule or reaction
        """
        import re  
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(text)]
        assert text == ''.join(tokens), 'Error when parsing {}'.format(text)
        return tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        assert isinstance(self.vocab, torchtext.vocab.Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.stoi[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        assert isinstance(self.vocab, torchtext.vocab.Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.itos[index]

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = "".join(tokens).strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A Mol sequence has the following format:
        - single sequence: ``<s> X </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        torch.save(self.vocab, vocab_path)


def data_collator(batch, pad_token_id):
    whole_batch = {}
    ex = batch[0]
    for key in ex.keys():
        if 'mask' in key:
            padding_value = 0
        else:
            padding_value = pad_token_id
        whole_batch[key] = pad_sequence([x[key] for x in batch],
                                        batch_first=True,
                                        padding_value=padding_value)
    source_ids, source_mask, y = \
        whole_batch["input_ids"], whole_batch["attention_mask"], whole_batch["decoder_input_ids"]
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone()
    lm_labels[y[:, 1:] == padding_value] = -100
    return {'input_ids': source_ids, 'attention_mask': source_mask,
            'decoder_input_ids': y_ids, 'labels': lm_labels}

class YieldDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        type_path="train",
        max_source_length=500,
        n_obs=None,
        sep_id=[2768],
        prefix="",
    ):
        super().__init__()
        # FIXME: the rstrip logic strips all the chars, it seems.
        tok_name = tokenizer.__class__.__name__.lower().rstrip("tokenizer")

        dataframe = pd.read_csv(data_path, sep='\s+', header=None)
        if sep_id[0] > 0:
            if type_path == 'train':
                dataframe = dataframe.iloc[:sep_id[0]]
            elif type_path == 'val':
                if len(sep_id) == 1:
                    dataframe = dataframe.iloc[sep_id[0]:]
                else:
                    dataframe = dataframe.iloc[sep_id[0]:sep_id[1]]
            else:
                if len(sep_id) == 1:
                    dataframe = dataframe.iloc[sep_id[0]:]
                else:
                    dataframe = dataframe.iloc[sep_id[1]:]
        self.source = []
        for text in tqdm(dataframe[0], desc=f"Tokenizing {data_path}"):
            tokenized = tokenizer(
                [text],
                max_length=max_source_length,
                padding="do_not_pad",
                truncation=True,
                return_tensors='pt',
            )
            self.source.append(tokenized)

        self.target = [float(x) for x in dataframe[1]]
        
        if n_obs is not None:
            self.source = self.source[:n_obs]
        self.bos_token_id = tokenizer.bos_token_id

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "decoder_input_ids": torch.LongTensor([self.bos_token_id]),
                "labels": torch.tensor([target_ids])}

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


def data_collator_yield(batch, pad_token_id, percentage=False):
    whole_batch = {}
    ex = batch[0]
    for key in ex.keys():
        if percentage and 'labels' in key:
            whole_batch[key] = torch.Tensor([x[key] for x in batch])/100
            continue
        if 'mask' in key:
            padding_value = 0
        else:
            padding_value = pad_token_id
        whole_batch[key] = pad_sequence([x[key] for x in batch],
                                        batch_first=True,
                                        padding_value=padding_value)
    return whole_batch