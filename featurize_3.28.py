'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

'''

from collections import defaultdict, Counter
import glob
import re
from stop_words import get_stop_words
import numpy as np
import pandas as pd
import scipy.io
import csv

cachedStopWords = get_stop_words('english') + ["co", "http", "https", "i", "my", 
                  "our", "we", "you", "amp", "t"]

NUM_TEST_EXAMPLES = 10000

BASE_DIR = './'
PROLIFE_DIR = 'prolife/'
PROCHOICE_DIR = 'prochoice/'
INDIVIDUAL_DIR = 'individual/'

# ************* Features *************

# Features that look for certain words

def freq_feature(text, freq, key_word):
    return float(freq[key_word])
# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
list_of_words = pd.read_csv("C:/Users/Andy Hyunouk Ko/Desktop/School/Probability&Statistics/"
                 "Stat198/TwitterAnalysis-master/TwitterAnalysis-master/feat_words.csv", names=['words'])
feat_words = list_of_words["words"].values.tolist()
prolife_bag = feat_words[0:100]
prochoice_bag = feat_words[100:200]
overlap = []
prolife_unique_bag =prolife_bag[:]
prochoice_unique_bag =prochoice_bag[:]
unique_feat_words = feat_words[:]

for word in prolife_bag:
    if word in prochoice_bag:
        overlap.append(word)

for word in prolife_bag:
    if word in overlap:
        prolife_unique_bag.remove(word)

for word in prochoice_bag:
    if word in overlap:
        prochoice_unique_bag.remove(word)

for word in feat_words:
    if word in overlap:
        unique_feat_words.remove(word)

# Generates a feature vector
def generate_feature_vector(text, freq, feat_words):
    feature = []
    key_words = feat_words
    for key_word in key_words:
        feature.append(freq_feature(text, freq, key_word))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    
    return feature


# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames, feat_words):
    design_matrix = []
    whole_freq = defaultdict(int)

    for filename in filenames:
        with open(filename, "r", encoding='utf-8', errors='ignore') as f:
            text = f.read() # Read in text from file
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word = word.lower()
                word_freq[word] += 1
                if word not in cachedStopWords:
                    whole_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq, feat_words)
            design_matrix.append(feature_vector)
    d = Counter(whole_freq)
    """ This part is for initial creation of feat_words file
    with open('temporary.csv', 'a') as f:
        writer = csv.writer(f)
        for k, v in d.most_common(100):
            print(k,v)
            writer.writerow([k])
    """
    return design_matrix


# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW
def feature_matrix_generator(feat_words, mat_name):
    prolife_filenames = glob.glob(BASE_DIR + PROLIFE_DIR + '*.txt')
    print("prolife-----------------")
    prolife_design_matrix = generate_design_matrix(prolife_filenames, feat_words)
    prochoice_filenames = glob.glob(BASE_DIR + PROCHOICE_DIR + '*.txt')
    print("prochoice---------------")
    prochoice_design_matrix = generate_design_matrix(prochoice_filenames, feat_words)
    individual_filenames = glob.glob(BASE_DIR + INDIVIDUAL_DIR + '*.txt')
    print("individual---------------")
    individual_design_matrix = generate_design_matrix(individual_filenames, feat_words)

    X = prolife_design_matrix + prochoice_design_matrix
    Y = [1]*len(prolife_design_matrix) + [0]*len(prochoice_design_matrix)
    file_dict = {}
    file_dict['training_data'] = X
    file_dict['training_labels'] = Y
    file_dict['individual_data'] = individual_design_matrix
    train_data = np.array(X)
    indiv_data = np.array(individual_design_matrix)
    indiv_total_occur = np.sum(indiv_data, axis=1)
    train_total_occur = np.sum(train_data, axis=1)
    file_dict['indiv_prop_mat'] = indiv_data / indiv_total_occur[:, None]
    file_dict['train_prop_mat'] = train_data / train_total_occur[:, None]
    scipy.io.savemat(mat_name+'.mat', file_dict, do_compression=True)
    return

feature_matrix_generator(feat_words, '2bags')
# Commented out code below is using overlapping bags which yielded insignificant results
"""
feature_matrix_generator(feat_words, '2bags')
feature_matrix_generator(prolife_bag, 'prolife_bag')
feature_matrix_generator(overlap, 'overlap')
"""

feature_matrix_generator(prolife_unique_bag, 'prolife_unique_bag')
feature_matrix_generator(unique_feat_words, 'unique_feat_words')

"""
#Calculate the stance of each of the users using bag of words
data = scipy.io.loadmat('2bags.mat')
data = data['individual_data']
df = np.array(data)
total_prolife_occurence = np.sum(df[:, 0:100], axis=1)
total_occurence = np.sum(df, axis=1)

data2 = scipy.io.loadmat('overlap.mat')
data2 = data2['individual_data']
df2 = np.array(data2)
overlap_occurence = np.sum(df2, axis=1)

M = (total_prolife_occurence - 0.5*overlap_occurence)/total_occurence
stance = 2*(0.5-M)
#there are 59 overlaps
"""

#Modified bags so that thee is no overlap, for individuals:
u_data = scipy.io.loadmat('unique_feat_words.mat')
u_data_indiv = u_data['individual_data']
u_data_train = u_data['training_data']
u_df_indiv = np.array(u_data_indiv)
u_df_train = np.array(u_data_train)

u_data2 = scipy.io.loadmat('prolife_unique_bag.mat')
u_data2_indiv = u_data2['individual_data']
u_data2_train = u_data2['training_data']
u_df2_indiv = np.array(u_data2_indiv)
u_df2_train = np.array(u_data2_train)

indiv_total_prolife_occurence = np.sum(u_df2_indiv, axis=1)
indiv_total_occurence = np.sum(u_df_indiv, axis=1)

train_total_prolife_occurence = np.sum(u_df2_train, axis=1)
train_total_occurence = np.sum(u_df_train, axis=1)

indiv_u_M = indiv_total_prolife_occurence/indiv_total_occurence
indiv_u_M[np.isnan(indiv_u_M)] = 0.5
indiv_u_stance = 2*(0.5-indiv_u_M)

train_u_M = train_total_prolife_occurence/train_total_occurence
train_u_M[np.isnan(train_u_M)] = 0.5
train_u_stance = 2*(0.5-train_u_M)


individual_filenames = glob.glob(BASE_DIR + INDIVIDUAL_DIR + '*.txt')
file_dict  = {}
file_dict['indiv_stance'] = indiv_u_stance
file_dict['train_stance'] = train_u_stance
file_dict['individual_account order'] = [s.split('/')[1].split('\\')[1].split('.')[0] for s in individual_filenames]
scipy.io.savemat('stance.mat', file_dict, do_compression=True)

'''
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = [1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)

file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_design_matrix
scipy.io.savemat('spam_data.mat', file_dict, do_compression=True)
'''

