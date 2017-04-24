import scipy.io
import pandas as pd
# Combine Stance
bow_data = scipy.io.loadmat('stance_word.mat')
classifier_data = scipy.io.loadmat('logistic_prop_word.mat')
indiv_df_1 = pd.DataFrame({
	'account_name': [s.strip() for s in bow_data['individual_account order']],
	'stance': bow_data['indiv_stance'].flatten()
	})
indiv_df_2 = pd.DataFrame({
	'User': [s.strip() for s in classifier_data['individual_account_order']],
	'prochoice_strength': classifier_data['indiv_prochoice_strength'].flatten()
	})
indiv_df = pd.merge(indiv_df_1, indiv_df_2, left_on='account_name', right_on='User', how='left')
indiv_df = indiv_df.drop('User', axis=1)
bow_stance = indiv_df['stance']
bow_stance = bow_stance

classifier_stance = indiv_df['prochoice_strength']
classifier_stance = classifier_stance
classifier_stance = 2*(classifier_stance-0.5)

final_stance = 0.5*(bow_stance + classifier_stance)
indiv_final_stance_word = pd.DataFrame({
	'User': [s.strip() for s in bow_data['individual_account order']],
	'stance': final_stance
	}) 
indiv_final_stance_word.to_csv('indiv_final_stance_word.csv', index=False)

# ------- compute final stance for organizational accounts with words--------- #
org_df_1 = pd.DataFrame({
	'account_name': [s.strip() for s in bow_data['known_account_order']],
	'stance': bow_data['train_stance'].flatten()
	})
org_df_2 = pd.DataFrame({
	'User': [s.strip() for s in classifier_data['org_account_order']],
	'prochoice_strength': classifier_data['org_prochoice_strength'].flatten()
	})
org_df = pd.merge(org_df_1, org_df_2, left_on='account_name', right_on='User', how='left')
org_df = org_df.drop('User', axis=1)
bow_stance = org_df['stance']
bow_stance = bow_stance

classifier_stance = org_df['prochoice_strength']
classifier_stance = classifier_stance
classifier_stance = 2*(classifier_stance-0.5)

final_stance = 0.5*(bow_stance + classifier_stance)
org_final_stance_word = pd.DataFrame({
	'User': [s.strip() for s in bow_data['known_account_order']],
	'stance': final_stance
	}) 
org_final_stance_word.to_csv('org_final_stance_word.csv', index=False)


# ------ Combine hashtag's stance --------#
bow_data = scipy.io.loadmat('stance_hashtag.mat')
classifier_data = scipy.io.loadmat('logistic_prop_hashtag.mat')
indiv_df_1 = pd.DataFrame({
	'account_name': [s.strip() for s in bow_data['individual_account order']],
	'stance': bow_data['indiv_stance'].flatten()
	})
indiv_df_2 = pd.DataFrame({
	'User': [s.strip() for s in classifier_data['individual_account_order']],
	'prochoice_strength': classifier_data['indiv_prochoice_strength'].flatten()
	})
indiv_df = pd.merge(indiv_df_1, indiv_df_2, left_on='account_name', right_on='User', how='left')
indiv_df = indiv_df.drop('User', axis=1)
bow_stance = indiv_df['stance']
bow_stance = bow_stance

classifier_stance = indiv_df['prochoice_strength']
classifier_stance = classifier_stance
classifier_stance = 2*(classifier_stance-0.5)

final_stance = 0.5*(bow_stance + classifier_stance)
indiv_final_stance_hashtag = pd.DataFrame({
	'User': [s.strip() for s in bow_data['individual_account order']],
	'stance': final_stance
	}) 
indiv_final_stance_hashtag.to_csv('indiv_final_stance_hashtag.csv', index=False)

# ------- compute final stance for organizational accounts with hashtags--------- #
org_df_1 = pd.DataFrame({
	'account_name': [s.strip() for s in bow_data['known_account_order']],
	'stance': bow_data['train_stance'].flatten()
	})
org_df_2 = pd.DataFrame({
	'User': [s.strip() for s in classifier_data['org_account_order']],
	'prochoice_strength': classifier_data['org_prochoice_strength'].flatten()
	})
org_df = pd.merge(org_df_1, org_df_2, left_on='account_name', right_on='User', how='left')
org_df = org_df.drop('User', axis=1)
bow_stance = org_df['stance']
bow_stance = bow_stance

classifier_stance = org_df['prochoice_strength']
classifier_stance = classifier_stance
classifier_stance = 2*(classifier_stance-0.5)

final_stance = 0.5*(bow_stance + classifier_stance)
org_final_stance_hashtag = pd.DataFrame({
	'User': [s.strip() for s in bow_data['known_account_order']],
	'stance': final_stance
	}) 
org_final_stance_hashtag.to_csv('org_final_stance_hashtag.csv', index=False)

'''
bow_data = scipy.io.loadmat('stance_hashtag.mat')
bow_stance = bow_data['indiv_stance']
bow_stance = bow_stance.flatten()

classifier_data = scipy.io.loadmat('indiv_logistic_prop_hashtag.mat')
classifier_stance = classifier_data['prochoice_strength']
classifier_stance = classifier_stance.flatten()
classifier_stance = 2*(classifier_stance-0.5)

final_stance = 0.5*(bow_stance + classifier_stance)
np.savetxt('final_stance_hashtag.csv', final_stance, delimiter=',')
'''