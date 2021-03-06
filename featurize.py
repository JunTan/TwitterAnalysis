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
import scipy.io
from stop_words import get_stop_words
from csv import DictReader

with open("select_word.csv") as f:
    words = [row["word"].strip() for row in DictReader(f)]

with open("select_hashtag.csv") as f:
    hashtags = [row["word"].strip() for row in DictReader(f)]



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

# Generates a feature vector
def generate_feature_vector_word(text, freq):
    feature = []
    
    for key_word in words:
        feature.append(freq_feature(text, freq, key_word))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    
    return feature

# Generates a feature vector
def generate_feature_vector_hashtag(text, freq):
    feature = []
    
    for key_word in hashtags:
        feature.append(freq_feature(text, freq, key_word))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    
    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames, output_wordcnt = True):
    design_matrix = []
    whole_freq = defaultdict(int)

    for filename in filenames:
        with open(filename, "r", encoding='utf-8', errors='ignore') as f:
            text = f.read() # Read in text from file
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            #words = re.findall(r"#(\w+)", text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word = word.lower()
                word_freq[word] += 1
                if word not in cachedStopWords:
                    whole_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector_word(text, word_freq)
            design_matrix.append(feature_vector)
    
    if output_wordcnt:
        d = Counter(whole_freq)
        # Create a feature vector
        for k, v in d.most_common(100):
            print (k, ",", v)
    
    return design_matrix

def generate_design_matrix_hashtag(filenames, output_wordcnt = True):
    design_matrix = []
    whole_freq = defaultdict(int)

    for filename in filenames:
        with open(filename, "r", encoding='utf-8', errors='ignore') as f:
            text = f.read() # Read in text from file
            text = text.replace('\r\n', ' ') # Remove newline character
            #words = re.findall(r'\w+', text)
            words = re.findall(r"#(\w+)", text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word = word.lower()
                word_freq[word] += 1
                if word not in cachedStopWords:
                    whole_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector_hashtag(text, word_freq)
            design_matrix.append(feature_vector)

    if output_wordcnt:
        d = Counter(whole_freq)
        # Create a feature vector
        for k, v in d.most_common(100):
            print (k, ",", v)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

print("word, count")
prolife_filenames = glob.glob(BASE_DIR + PROLIFE_DIR + '*.txt')
prolife_design_matrix = generate_design_matrix(prolife_filenames)

prochoice_filenames = glob.glob(BASE_DIR + PROCHOICE_DIR + '*.txt')
prochoice_design_matrix = generate_design_matrix(prochoice_filenames)

inidividual_filenames = glob.glob(BASE_DIR + INDIVIDUAL_DIR + '*.txt')
inidividual_design_matrix = generate_design_matrix(inidividual_filenames, False)

X = prolife_design_matrix + prochoice_design_matrix
Y = [1]*len(prolife_design_matrix) + [0]*len(prochoice_design_matrix)
file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['individual_data'] = inidividual_design_matrix
print(len(inidividual_design_matrix))
file_dict['individual_account_order'] = [s.split('/')[2].split('.')[0] for s in inidividual_filenames]
file_dict['known_account_order'] = [s.split('/')[2].split('.')[0] for s in prolife_filenames] +\
                                   [s.split('/')[2].split('.')[0] for s in prochoice_filenames]
scipy.io.savemat('word_vec.mat', file_dict, do_compression=True)


print("hashtag, count")
prolife_filenames = glob.glob(BASE_DIR + PROLIFE_DIR + '*.txt')
prolife_design_matrix = generate_design_matrix_hashtag(prolife_filenames)

prochoice_filenames = glob.glob(BASE_DIR + PROCHOICE_DIR + '*.txt')
prochoice_design_matrix = generate_design_matrix_hashtag(prochoice_filenames)

inidividual_filenames = glob.glob(BASE_DIR + INDIVIDUAL_DIR + '*.txt')
inidividual_design_matrix = generate_design_matrix_hashtag(inidividual_filenames, False)

X = prolife_design_matrix + prochoice_design_matrix
Y = [1]*len(prolife_design_matrix) + [0]*len(prochoice_design_matrix)
file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['individual_data'] = inidividual_design_matrix
file_dict['individual_account_order'] = [s.split('/')[2].split('.')[0] for s in inidividual_filenames]
file_dict['known_account_order'] = [s.split('/')[2].split('.')[0] for s in prolife_filenames] +\
                                   [s.split('/')[2].split('.')[0] for s in prochoice_filenames]
scipy.io.savemat('hashtag_vec.mat', file_dict, do_compression=True)


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

