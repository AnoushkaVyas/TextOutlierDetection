from base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from utils.text_encoders import MyBertTokenizer
from utils.misc import clean_text
from .preprocessing import compute_tfidf_weights


import torch
import nltk
import random

class Newsgroups20_Dataset(TorchnlpDataset):

    def __init__(self, root: str, outlier_class=0, tokenizer='spacy', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, clean_txt=False):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        normal=[]
        for i in range(4):
            if i != outlier_class:
                normal.append(i)

        groups = [
            ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
             'comp.windows.x'],
            ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
            ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
            ['misc.forsale'],
            ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
            ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']
        ]

        ids= {'comp.graphics':0 , 'comp.os.ms-windows.misc':0, 'comp.sys.ibm.pc.hardware':0, 'comp.sys.mac.hardware':0,
             'comp.windows.x':0, 'rec.autos':1, 'rec.motorcycles':1, 'rec.sport.baseball':1, 'rec.sport.hockey':1,
             'sci.crypt':2, 'sci.electronics':2, 'sci.med':2, 'sci.space':2,
             'misc.forsale':3}
        

        self.normal_classes = []
        for i in normal:
            self.normal_classes += groups[i]

        outlier=[outlier_class,4,5]
        self.outlier_classes=[]
        for i in range(3):
            self.outlier_classes += groups[outlier[i]]

        # Load the 20 Newsgroups dataset
        self.train_set, self.test_set = newsgroups20_dataset(directory=root, train=True, test=True, clean_txt=clean_txt)

        self.data=self.train_set

        # Pre-process
        self.data.columns.add('index')
        self.data.columns.add('classlabel')
        self.data.columns.add('weight')

        data_idx = []  # for subsetting dataset
        count_normal=0
        for i, row in enumerate(self.data):
            if row['label'] in self.normal_classes:
                data_idx.append(i)
                row['classlabel']=torch.tensor(ids[row['label']])
                row['label'] = torch.tensor(0)
                count_normal=count_normal+1
                
            row['text'] = row['text'].lower()

        number_of_outliers= int((0.01*count_normal)/0.99)

        for i in range(number_of_outliers):
            outlier_idx=[]
            outlier_class_name=random.sample(self.outlier_classes,1)
            for j, row in enumerate(self.data):
                if j not in data_idx:
                    if outlier_class_name[0] == row['label']:
                        outlier_idx.append(j)
        
            if len(outlier_idx) > 0:
                index=random.sample(outlier_idx,1)[0]
                self.data[index]['label']=torch.tensor(1)
                self.data[index]['classlabel']=torch.tensor(4)
                data_idx.append(index)

        # Subset dataset
        self.data = Subset(self.data, data_idx)

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
            print(row['text'].shape)
            row['index'] = i
    
def newsgroups20_dataset(directory, train, test, clean_txt):
    """
    Load the 20 Newsgroups dataset.

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train and test) depending on if their respective boolean argument
        is ``True``.
    """

    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    for split_set in splits:

        dataset = fetch_20newsgroups(data_home=directory, subset=split_set, remove=('headers', 'footers', 'quotes'))
        examples = []

        for id in range(len(dataset.data)):
            if clean_txt:
                text = clean_text(dataset.data[id])
            else:
                text = ' '.join(word_tokenize(dataset.data[id]))
            label = dataset.target_names[int(dataset.target[id])]

            if text:
                examples.append({
                    'text': text,
                    'label': label
                })

        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
