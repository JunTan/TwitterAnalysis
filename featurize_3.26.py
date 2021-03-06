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
df = pd.read_csv("C:/Users/Andy Hyunouk Ko/Desktop/School/Probability&Statistics/"
                 "Stat198/TwitterAnalysis-master/TwitterAnalysis-master/feat_words.csv", names=['words'])
feat_words = df["words"].values.tolist()
prolife_bag = feat_words[0:100]
prochoice_bag = feat_words[100:200]
overlap = []
for word in prolife_bag:
    if word in prochoice_bag:
        overlap.append(word)

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
    """ This part is intended for initial creation of feature vectors; hence is run only once
    with open('feat_words.csv', 'a') as f:
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

# Each of these lines below generates a compressed matrix using 3 different bags:
# 1. Both prolife and prochoice bags combined, 2. prolife bag only, 3. prochoice bag only
feature_matrix_generator(feat_words, '2bags')
feature_matrix_generator(prolife_bag, 'prolife_bag')
feature_matrix_generator(prochoice_bag, 'prochoice_bag')
feature_matrix_generator(overlap, 'overlap')

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

