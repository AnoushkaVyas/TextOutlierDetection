import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from cvdd import CVDD


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['reuters', 'newsgroups20',]))
@click.argument('net_name', type=click.Choice(['cvdd_Net']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=42, help='Set seed. If -1, use randomization.')
@click.option('--tokenizer', default='spacy', type=click.Choice(['spacy', 'bert']), help='Select text tokenizer.')
@click.option('--clean_txt', is_flag=True, help='Specify if text should be cleaned in a pre-processing step.')
@click.option('--embedding_size', type=int, default=300, help='Size of the word vector embedding.')
@click.option('--pretrained_model', default='GloVe_6B',
              type=click.Choice([None, 'GloVe_6B', 'GloVe_42B', 'GloVe_840B', 'GloVe_twitter.27B', 'FastText_en',
                                 'bert']),
              help='Load pre-trained word vectors or language models to initialize the word embeddings.')
@click.option('--n_attention_heads', type=int, default=3, help='Number of attention heads in self-attention module.')
@click.option('--attention_size', type=int, default=100, help='Self-attention module dimensionality.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--n_threads', type=int, default=0,
              help='Sets the number of OpenMP threads used for parallelizing CPU operations')
@click.option('--outlier_class', type=int, default=0,
              help='Specify the outlier class of the dataset (all other classes are considered anomalous).')
@click.option('--clusters', type=int, default=4,
              help='Number of clusters in regular data')

def main(dataset_name, net_name, xp_path, data_path, load_model, device, seed, tokenizer, clean_txt,
         embedding_size, pretrained_model, n_attention_heads, attention_size, n_jobs_dataloader, n_threads,
         outlier_class,clusters):
    """
    Context Vector Data Description (CVDD): An unsupervised anomaly detection method for text.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Outlier class: %d' % outlier_class)
    logger.info('Cluster size: %d' % clusters)
    logger.info('Network: %s' % net_name)
    logger.info('Tokenizer: %s' % cfg.settings['tokenizer'])
    logger.info('Clean text in pre-processing: %s' % cfg.settings['clean_txt'])
    if cfg.settings['embedding_size'] is not None:
        logger.info('Word vector embedding size: %d' % cfg.settings['embedding_size'])
    logger.info('Load pre-trained model: %s' % cfg.settings['pretrained_model'])

    # Print CVDD configuration)
    logger.info('Number of attention heads: %d' % cfg.settings['n_attention_heads'])
    logger.info('Attention size: %d' % cfg.settings['attention_size'])

    
    # Set seed for reproducibility
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)
    if n_threads > 0:
        torch.set_num_threads(n_threads)
        logger.info('Number of threads used for parallelizing CPU operations: %d' % n_threads)

    # Load data
    dataset =torch.load(data_path)

    # Initialize CVDD model and set word embedding
    cvdd = CVDD()
    cvdd.set_network(net_name=net_name,
                     dataset=dataset,
                     pretrained_model=cfg.settings['pretrained_model'],
                     embedding_size=cfg.settings['embedding_size'],
                     attention_size=cfg.settings['attention_size'],
                     n_attention_heads=cfg.settings['n_attention_heads'],
                     clusters=cfg.settings['clusters'])

    # If specified, load model parameters from already trained model
    if load_model:
        cvdd.load_model(import_path=load_model, device=device)
        logger.info('Loading model from %s.' % load_model)

    # Train model on dataset
    cvdd.train(dataset,
               device=device,
               n_jobs_dataloader=n_jobs_dataloader)

    # Save Model
    cvdd.save_model(export_path=xp_path + '/model.tar')

if __name__ == '__main__':
    main()
