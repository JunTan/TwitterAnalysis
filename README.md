# Package needed

`pip install stop-words`

`pip install scipy`

# Featurize.py
This file generates a design matrix for prolife group, prochoice group and individual group and stores it in the 'abortion.mat'. Running this file also outputs a list of most frequent words from prolife and prochoice group. To avoid the output flushs your screen, you can redirect the output to a file (eg. word.txt) by doing 

`python featurize.py > word.txt`

Each row of the design matrix corresponding a sample(user). Each column of the design matrix is the word frequency of a particular word.

To change the features, change the `key_words` list in the `generate_feature_vector` function

