## Installation
This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`.

Clone the repository to your machine and directory of choice.

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `virtualenv`
```
# pip install virtualenv
cd <path-to-CVDD-PyTorch-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-CVDD-PyTorch-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```

After installing the packages, run `python -m spacy download en` to download the [spaCy](https://spacy.io/) `en` 
library.


## Running experiments
You can run CVDD experiments using the `main.py` script.

The following are examples on how to run experiments on 
[`Reuters-21578`](http://www.daviddlewis.com/resources/testcollections/reuters21578/), 
[`20 Newsgroups`](http://qwone.com/~jason/20Newsgroups/), and 
[`IMDB Movie Reviews`](http://ai.stanford.edu/~amaas/data/sentiment/) as reported in the paper.

### Reuters-21578
 ```
cd <path-to-CVDD-PyTorch-directory>
```
# activate virtual environment

```
source myenv/bin/activate  # or 'source activate myenv' for conda
```

# change to source directory
```
cd src
```

# run experiment
```
python3 main.py reuters cvdd_Net ../log/test_reuters ../data/reuters0.pt
```
