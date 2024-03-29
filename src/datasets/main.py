from .reuters21578 import Reuters_Dataset
from .newsgroups20 import Newsgroups20_Dataset

def load_dataset(dataset_name, data_path, outlier_class, tokenizer='spacy', use_tfidf_weights=False,
                 append_sos=False, append_eos=False, clean_txt=False):
    """Loads the dataset."""

    implemented_datasets = ('reuters', 'newsgroups20')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'reuters':
        dataset = Reuters_Dataset(root=data_path, outlier_class=outlier_class, tokenizer=tokenizer,
                                  use_tfidf_weights=use_tfidf_weights, append_sos=append_sos, append_eos=append_eos,
                                  clean_txt=clean_txt)

    if dataset_name == 'newsgroups20':
        dataset = Newsgroups20_Dataset(root=data_path, outlier_class=outlier_class, tokenizer=tokenizer,
                                       use_tfidf_weights=use_tfidf_weights, append_sos=append_sos,
                                       append_eos=append_eos, clean_txt=clean_txt)

    return dataset
