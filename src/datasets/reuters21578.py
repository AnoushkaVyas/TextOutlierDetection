import torch
import nltk
import random


from base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from nltk.corpus import reuters
from nltk import word_tokenize
from utils.text_encoders import MyBertTokenizer
from utils.misc import clean_text
from .preprocessing import compute_tfidf_weights

class Reuters_Dataset(TorchnlpDataset):

    def __init__(self, root: str, outlier_class=0, tokenizer='spacy', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, clean_txt=False):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        classes = ['earn', 'acq', 'money-fx', 'grain', 'crude']

        self.outlier_classes = [classes[outlier_class]]
        del classes[outlier_class]
        self.normal_classes = classes

        # Load the reuters dataset
        self.train_set, self.test_set = reuters_dataset(directory=root, train=True, test=True, clean_txt=clean_txt)

        self.data=self.train_set+self.test_set

        # Pre-process
        self.data.columns.add('index')
        self.data.columns.add('weight')

        data_idx= []  # for subsetting dataset
        outlier_idx=[] 
        count_normal=0
        for i, row in enumerate(self.data):
            if any(label in self.normal_classes for label in row['label']) and (len(row['label']) == 1):
                data_idx.append(i)
                row['label'] = torch.tensor(0)
                count_normal=count_normal+1

            elif any(label in self.outlier_classes for label in row['label']) and (len(row['label']) == 1):
                outlier_idx.append(i)

            row['text'] = row['text'].lower()

        number_of_outliers= int(0.05*count_normal)
        
        outlier_idx=sorted(random.sample(outlier_idx,number_of_outliers))

        for i in range(len(outlier_idx)):
            self.data[outlier_idx[i]]['label']=torch.tensor(1)
            data_idx.append(outlier_idx[i])

        # Subset dataset to a few classes
        self.data = Subset(self.data, sorted(data_idx))

        # Make corpus and set encoder
        text_corpus = [row['text'] for row in datasets_iterator(self.data)]
        
        if tokenizer == 'spacy':
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos)
        if tokenizer == 'bert':
            self.encoder = MyBertTokenizer.from_pretrained('bert-base-uncased', cache_dir=root)

        # Encode
        for row in datasets_iterator(self.data):
            if append_sos:
                sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), self.encoder.encode(row['text'])))
            else:
                row['text'] = self.encoder.encode(row['text'])

        # Compute tf-idf weights
        if use_tfidf_weights:
            compute_tfidf_weights(self.data, vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.data):
                row['weight'] = torch.empty(0)

        # Get indices after pre-processing
        for i, row in enumerate(self.data):
            row['index'] = i

def reuters_dataset(directory='../data', train=True, test=False, clean_txt=False):
    """
    Load the Reuters-21578 dataset.

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train and test) depending on if their respective boolean argument
        is ``True``.
    """

    nltk.download('reuters', download_dir=directory)
    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    doc_ids = reuters.fileids()

    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    for split_set in splits:

        split_set_doc_ids = list(filter(lambda doc: doc.startswith(split_set), doc_ids))
        examples = []

        for id in split_set_doc_ids:
            if clean_txt:
                text = clean_text(reuters.raw(id))
            else:
                text = ' '.join(word_tokenize(reuters.raw(id)))
            labels = reuters.categories(id)

            examples.append({
                'text': text,
                'label': labels,
            })

        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
