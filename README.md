# Replication of IBM Team's Duplicate Question Detection Experiment

## Abstract

Validation of experimental results through their replication is central to the scientific progress, in particular in cases that may represent important breakthroughs with respect to the state of the art.
In the present paper we report on the exercise we undertook to replicate the central result of the experiment reported in the (Bogdanova et al., 2015) paper, *Detecting Semantically Equivalent Questions in Online User Forums*, which achieved results far surpassing the state-of-the-art for the task of duplicate question detection. In particular, we report on how our exercise allowed to find a flaw in the preparation of the data used in that paper that casts justified doubt on the validity of the breakthrough results reported there.

## Our paper

We presented our replication exercise in this paper: 

João Silva, João Rodrigues, Vladislav Maraev, Chakaveh Saedi and António Branco, 2018, "A 20% Jump in Duplicate Question Detection Accuracy? Replicating IBM team's experiment and finding problems in its data preparation". *In Proceedings of Workshop on Replicability and Reproducibility of Research Results in Science and Technology of Language (4REAL2018)*, Colocated with LREC2018, Miyazaki, Japan, May 12, 2018.

## IBM team's paper

This is the paper reporting the IBM team's work:

Dasha Bogdanova, Cícero Nogueira dos Santos, Luciano Barbosa and Bianca Zadrozny, 2015,
"Detecting Semantically Equivalent Questions in Online User Forums",
In *Proceedings of the 19th Conference on Computational Natural Language Learning (CoNLL2015)*,
Colocated with ACL2015,
Beijing, China, 30-31 July, 2015,
pp. 123-131,
http://aclweb.org/anthology/K/K15/K15-1013.pdf

## Data and code used for the replication

### Prerequisites
1. Download the source code from http://lxcenter.di.fc.ul.pt/datasets/msrdsdl/msrdsdl.tar.gz .

2. Extract the code:

     `tar -xvf msrdsdl.tar.gz`
   
3. You will need *Python version 3.4.3 or higher*.

4. Install required packages:

     `pip install -r requirements.txt`

5. Set up Theano backend for Keras by editing the configuration file `~/.keras/keras.json` and changing the field `backend` to `theano`.

### About the program

After satisfying all the prerequisites you will have the following directory structure: 

```
|-- cnn.py
|-- data
|   |-- askubuntu
|   |   |-- clue
|   |   |   |-- test.tsv
|   |   |   |-- train.tsv
|   |   |   '-- val.tsv
|   |   '-- noclue
|   |       |-- test.tsv
|   |       |-- train.tsv
|   |       '-- val.tsv
|   |-- meta
|   |   |-- clue
|   |   |   |-- test.tsv
|   |   |   |-- train.tsv
|   |   |   '-- val.tsv
|   |   '-- noclue
|   |       |-- test.tsv
|   |       |-- train.tsv
|   |       '-- val.tsv
|-- __init.py__
|-- models
|   |-- askubuntu.w2v
|   |-- meta.w2v
|-- preprocess.py
'-- requirements.txt
```

You can run the application `./cnn.py` with the `--help` argument to see available parameters.

To change hyperparameters you can modify the method `SentenceSimilarity. set_hyperparameters` by adding new modes. 

## Question Answering
### Replication of the work by Bogdanova et al. (2015)
    
For AskUbuntu dataset:
    
```
  ./cnn.py replication --train data/askubuntu/clue/train.tsv \
           --val data/askubuntu/clue/val.tsv \
           --test data/askubuntu/clue/test.tsv \
           --w2v models/askubuntu.w2v
```
    
For META Stackexchange dataset:

```
./cnn.py replication --train data/meta/clue/train.tsv \
         --val data/meta/clue/val.tsv \
         --test data/meta/clue/test.tsv \
         --w2v models/meta.w2v
```

### Impact of text preprocessing (clue phrases removed)

For AskUbuntu dataset:

```
./cnn.py pp_impact --train data/askubuntu/noclue/train.tsv \
         --val data/askubuntu/noclue/val.tsv \
         --test data/askubuntu/noclue/test.tsv \
         --w2v models/askubuntu.w2v
```

For META Stackexchange dataset: 

```
./cnn.py pp_impact --train data/meta/noclue/train.tsv \
         --val data/meta/noclue/val.tsv \
         --test data/meta/noclue/test.tsv \
         --w2v models/meta.w2v
```

### Impact of word embeddings (no pre-trained word embeddings)

```
./cnn.py we_impact --train data/askubuntu/noclue/train.tsv \
         --val data/askubuntu/noclue/val.tsv \
         --test data/askubuntu/noclue/test.tsv
```
