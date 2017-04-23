import scipy.io
import numpy as np
#Combine Stance
bow_data = scipy.io.loadmat('stance.mat')
bow_stance = bow_data['indiv_stance']
bow_stance = bow_stance.flatten()


classifier_data = scipy.io.loadmat('indiv_logistic_prop_word.mat')
classifier_stance = classifier_data['prochoice_strength']
classifier_stance = classifier_stance.flatten()
classifier_stance = 2*(classifier_stance-0.5)

final_stance = 0.5*(bow_stance + classifier_stance)
np.savetxt('final_stance_word.csv', final_stance, delimiter=',')

classifier_data = scipy.io.loadmat('indiv_logistic_prop_hashtag.mat')
classifier_stance = classifier_data['prochoice_strength']
classifier_stance = classifier_stance.flatten()
classifier_stance = 2*(classifier_stance-0.5)

final_stance = 0.5*(bow_stance + classifier_stance)
np.savetxt('final_stance_hashtag.csv', final_stance, delimiter=',')