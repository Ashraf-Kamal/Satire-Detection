# Proposed BiSAT model
--------------------------------
The proposed BiSAT model with an input layer to process the input text and follows on to an embedding layer to embed the text into a numeric representation. The first BiLSTM process sequences in both forward and backward directions, enabling them to capture information from past and future time steps simultaneously. This helps in understanding satirical context in both directions. Thereafter, self-attention layer is responsible to calculate attention weights between different elements of the sequence, enabling the model to focus on relevant satirical information and facilitate long-range dependencies without recurrence. The second BiLSM receives the outcome of the self-attention layer and it helps in capturing more complex satirical patterns and dependencies in the sequences. In addtion to that four groups of hand-crafted shallow which consist of 13 auxiliary features making the model well-informed. The proposed model is evaluated over two benchmark datasets and a newly created Satire-280 dataset. It performs significantly better than the comparable methods.

# Pre-requisite:
--------------------------------

1. Twitter REST API
2. Keras 2.2.4
3. Numpy
4. Pandas
5. Python 3.7, 2.7
6. GloVe
7. Tensorflow 1.15
8. NLTK

# Satire-280 Dataset
--------------------------------
Satire is one of the important categories of figurative language. This repository also contains a newly created "Satire-280" dataset. Due to the Twitter policy, only tweet ids and their respective labels are shared for "Satire-280" dataset. This dataset is crawled using Twitter REST API in Python by applying the hashtag-based annotation technique criteria. All tweets are based on Twitter's 280 characters limit, wherein satire-based tweets are collected using #satire hashtag, whereas non-satire tweets are collected using the hashtags, like #not, #love, and #hate. 

A short statistics of the two files of Satire-280 dataset is given below:-

1. Satire.xlsx: This file contains a total of 16374 tweet ids for satire category. 

2. Non_Satire.xlsx: This file contains a total of 25821 tweet ids for non-satire category.




