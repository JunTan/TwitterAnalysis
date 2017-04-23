# File and Output
`select_word.csv`: words for bag of word method

`select_hashtag.csv`: words for counting the hashtag

`featurize.py`: Generate the design matrix counting different words: bag of word and hashtags for prolife, prochoice and individuals
 
 output: word_vec.mat --- contains the design matrix counting the bag of word 

         hashtag_vec.mat --- contains the design matrix counting the hashtag

`Diversity_of_accounts`: Compute the following proportion of the individuals

output: prolife_proportion.csv, prochoice_prportion.csv, proportion.csv (contains the proportion of the individuals only)

`Classifier.ipynb`: Use logisitc regression to compute the likelihood of individuals belong to prolife or prochoice group

output: indiv_logistic_prop_word.mat, indivi_logistic_prop_hashtag.mat: probability output the logistic classifier

        indiv_strength_diversity.csv, indiv_strength_diversity_hashtag.csv: combination of the following proportion and the output of the logistic regression

`compute_final_stance.py`: compute the final stance of individuals

output: final_stance_word.mat, final_stance_hashtag.mat

`Sample followers.ipynb`: randomly samples followers from prolife or prochoice group

output: prolife_followers.csv, prochoice_followers.csv


# Package needed

`pip install stop-words`

`pip install scipy`

# Featurize.py
This file generates a design matrix for prolife group, prochoice group and individual group and stores it in the 'abortion.mat'. Running this file also outputs a list of most frequent words from prolife and prochoice group. To avoid the output flushs your screen, you can redirect the output to a file (eg. word.txt) by doing 

`python featurize.py > word.txt`

Each row of the design matrix corresponding a sample(user). Each column of the design matrix is the word frequency of a particular word.

To change the features, change the `key_words` list in the `generate_feature_vector` function

# Read .mat file
```
import scipy.io as sio

data_path = 'abortion.mat'

data = sio.loadmat(train_data_path)

data['training_data'] #load the training_data
```
