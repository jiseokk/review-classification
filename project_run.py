import project_tool_functions as p1
import utils

#-------------------------------------------------------------------------------
# Data loading
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = p1.bag_of_words(train_texts)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)
#
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------

toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')

T = 5
L = 10

thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
thetas_avg_pa = p1.average_passive_aggressive(toy_features, toy_labels, T, L)

def plot_toy_results(algo_name, thetas):
    utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

plot_toy_results('Perceptron', thetas_perceptron)
plot_toy_results('Average Perceptron', thetas_avg_perceptron)
plot_toy_results('Average Passive-Aggressive', thetas_avg_pa)

#-------------------------------------------------------------------------------
#
#
#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------

T = 4
L = 27

pct_train_accuracy, pct_val_accuracy = \
   p1.perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

avg_pct_train_accuracy, avg_pct_val_accuracy = \
   p1.average_perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

avg_pa_train_accuracy, avg_pa_val_accuracy = \
   p1.average_passive_aggressive_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
print("{:50} {:.4f}".format("Training accuracy for average passive-aggressive:", avg_pa_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for average passive-aggressive:", avg_pa_val_accuracy))


##-------------------------------------------------------------------------------
#
#
#-------------------------------------------------------------------------------
# Section 2.10
#-------------------------------------------------------------------------------

data = (train_bow_features, train_labels, val_bow_features, val_labels)

# values of T and lambda to try
Ts = [1,2,3,4,5,6,7,10,12,15,17,20,25]
Ls = [1,10,15,20,25,27,30,32,34,50]

#pct_tune_results = utils.tune_perceptron(Ts, *data)
#avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)

# fix values for L and T while tuning passive-aggressive T and L, respective
best_L = 27
best_T = 4

avg_pa_tune_results_T = utils.tune_passive_aggressive_T(best_L, Ts, *data)
avg_pa_tune_results_L = utils.tune_passive_aggressive_L(best_T, Ls, *data)

#utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
#utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
utils.plot_tune_results('Avg Passive-Aggressive', 'T', Ts, *avg_pa_tune_results_T)
utils.plot_tune_results('Avg Passive-Aggressive', 'L', Ls, *avg_pa_tune_results_L)


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
#
#
# Accuracy Computation
#-------------------------------------------------------------------------------

print "(train accuracy, test accuracy) before modification"
print p1.average_passive_aggressive_accuracy(train_bow_features,test_bow_features,train_labels,test_labels,T,L)

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# 
#
# Assign to best_theta, the weights (and not the bias!) learned by the most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------

best_theta = p1.average_passive_aggressive(test_bow_features, test_labels, best_T, best_L)[0]
wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
print("Most Explanatory Word Features")
print(sorted_word_features[:10])

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# 
#
# Assessing performance on the validation set.
# 
#-------------------------------------------------------------------------------
dictionary_mod = p1.modified_bag_of_words(train_texts)

train_final_features = p1.extract_final_features(train_texts, dictionary_mod)
val_final_features   = p1.extract_final_features(val_texts, dictionary_mod)
test_final_features  = p1.extract_final_features(test_texts, dictionary_mod)

data = (train_final_features, train_labels, val_final_features, val_labels)

# values of T and lambda to try
Ts = [1,2,3,4,5,6,7,10,12,15,17,20,25]
Ls = [1,10,15,20,25,32,38,42,44,47,50,55,60,70,80,100]

#pct_tune_results = utils.tune_perceptron(Ts, *data)
#avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)

# fix values for L and T while tuning passive-aggressive T and L, respective
best_L = 50
best_T = 2

#avg_pa_tune_results_T = utils.tune_passive_aggressive_T(best_L, Ts, *data)
#avg_pa_tune_results_L = utils.tune_passive_aggressive_L(best_T, Ls, *data)

#utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
#utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
#utils.plot_tune_results('Avg Passive-Aggressive', 'T', Ts, *avg_pa_tune_results_T)
#utils.plot_tune_results('Avg Passive-Aggressive', 'L', Ls, *avg_pa_tune_results_L)

print "(train accuracy, test accuracy) after modification"
print p1.average_passive_aggressive_accuracy(train_final_features,test_final_features,train_labels,test_labels,best_T,best_L)

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
#
#
#-------------------------------------------------------------------------------

submit_texts = [sample['text'] for sample in utils.load_data('reviews_submit.tsv')]

# 1. Extract the preferred features from the train and submit data
dictionary = p1.modified_bag_of_words(submit_texts)
train_final_features = p1.extract_final_features(train_texts, dictionary)
submit_final_features = p1.extract_final_features(submit_texts, dictionary)

# 2. Train the most accurate classifier
final_thetas = p1.average_passive_aggressive(train_final_features, train_labels, 2,50)

# 3. Classify and write out the submit predictions.
submit_predictions = p1.classify(submit_final_features, *final_thetas)
utils.write_predictions('reviews_submit.tsv', submit_predictions)

#-------------------------------------------------------------------------------
