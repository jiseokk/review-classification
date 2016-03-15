from string import punctuation, digits

import numpy as np
import matplotlib.pyplot as plt
import time

### Parameters
T_perceptron = 5

T_avperceptron = 5

T_avgpa = 5
L_avgpa = 10
###

### Part I

feat = [[1., 2.], [1., 0.]]
labels = [1., -1.]
theta = [1.,1.]
theta_0 = -2.1



def hinge_loss(feature_matrix, labels, theta, theta_0):
    product = np.dot(feature_matrix, theta)   
    offset = np.zeros(len(feature_matrix))  #Create empty array for Theta_0
    offset.fill(float(theta_0))  #Fill all space with Theta_0
    comp = np.add(product, offset) #Th_0
    res = np.multiply(comp, labels) #y(Theta*x + Theta_0)
    
    #modify to the actual form
    for i in range(len(res)):  
        res[i] = max(0, 1-res[i])            
    
    return np.mean(res)
   

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    adj_feature = np.multiply(label,feature_vector) #yx
    new_theta = np.add(current_theta, adj_feature) #Theta + yx
    new_theta_0 = current_theta_0 + label #Theta_0' = Theta_0 + y
    return (new_theta, new_theta_0)
    

def perceptron(feature_matrix, labels, T):  
    theta = np.empty_like(feature_matrix[0]) #Set up theta, same dimension as feature veector
    theta.fill(0.) #Fill Theta with zeros
    theta_0 = 0.0 #Initialize Theta_0 to zero
    ticker = 0 #Initialize number to track the number of iterations to feature vector set
    
    
    while ticker < T:
        for i in range(len(feature_matrix)): #iterating through teh whole feature matrix
            check_before_label = np.add(np.dot(theta, feature_matrix[i]),theta_0)
            
            check_mult_label = np.multiply(labels[i], check_before_label)
            if check_mult_label == 0 or check_mult_label < 0:                
                (theta, theta_0) = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)

        ticker += 1

    return (theta, theta_0)
    
    

def passive_aggressive_single_step_update(feature_vector, label, L, current_theta, current_theta_0):
    product = np.dot(feature_vector, current_theta) #Theta*x
    offset = 0.0  #Create empty array for Theta_0  #Fill all space with Theta_0
    offset = current_theta_0
    comp = np.add(product, offset) #Theta*x+Theta_0
    result = np.multiply(comp, label) #y(Theta*x+Theta_0)

    hinge_loss = max(0,1 - result) 
    
    hinge_term = hinge_loss/np.dot(feature_vector, feature_vector)
    
    eta = min(1/float(L), hinge_term) #if hinge_loss is zero, eta is automatically zero and no update made   
 
    
    adj_feature = np.multiply(label,feature_vector) #y*x
    adj_eta_feature = np.multiply(eta,adj_feature) #n*y*x

    new_theta = np.add(current_theta, adj_eta_feature) #Theta_k + nyx
    new_theta_0 = current_theta_0 + label*eta 
    return (new_theta, new_theta_0)    
    
def average_perceptron(feature_matrix, labels, T):
    theta = np.empty_like(feature_matrix[0])
    theta.fill(0.)
    theta_sum = theta  
    theta_0 = 0.0
    theta_0_sum = theta_0
    ticker = 0
    update_track = 0
    
    while ticker < T:
        
        for i in range(len(feature_matrix)):        

            check_before_label = np.add(np.dot(theta, feature_matrix[i]),theta_0)

            check_mult_label = np.multiply(labels[i], check_before_label)
            if check_mult_label == 0 or check_mult_label < 0:
                update_track += 1                
                (theta, theta_0) = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
                theta_sum = np.add(theta, theta_sum)
                theta_0_sum += theta_0

        ticker += 1
        
    theta_average = np.divide(theta_sum, update_track)
    theta_0_average = theta_0_sum/update_track
        
    return (theta_average, theta_0_average)


def average_passive_aggressive(feature_matrix, labels, T, L):    
    theta = np.empty_like(feature_matrix[0])
    theta.fill(0.)
    theta_empty = np.empty_like(feature_matrix[0])
    theta_empty.fill(0.)
    theta_sum = theta  
    theta_0 = 0.0
    theta_0_sum = theta_0
    ticker = 0
    update_track = 0
    
    while ticker < T:
        
        for i in range(len(feature_matrix)):
  
            (theta_new, theta_0_new) = passive_aggressive_single_step_update(feature_matrix[i], labels[i], L, theta, theta_0)
                      

            if np.any(np.subtract(theta_new, theta)) or theta_0_new - theta_0 != 0: #Select for the instances where the theta actually gets updated
                theta_sum = np.add(theta_new, theta_sum)
                theta_0_sum += theta_0_new                
                update_track += 1
                theta = theta_new
                theta_0 = theta_0_new
            

        ticker += 1
        
    theta_average = np.divide(theta_sum, update_track)
    theta_0_average = theta_0_sum/update_track

    return (theta_average, theta_0_average)
    

### Part II

def classify(feature_matrix, theta, theta_0):
    product = np.dot(feature_matrix, theta) #Theta*x
    offset = np.arange(len(feature_matrix))  #Create empty array for Theta_0
    offset.fill(float(theta_0))
    output = np.add(product, offset) #Theta*x + Theta_0
    return np.sign(output)


    

def perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    (theta_train, theta_0_train) = perceptron(train_feature_matrix, train_labels, T)    
    train_preds = classify(train_feature_matrix, theta_train, theta_0_train)
    train_targets = train_labels 
    train_acc = accuracy(train_preds, train_targets)   
    
    val_preds = classify(val_feature_matrix, theta_train, theta_0_train)
    val_targets = val_labels
    val_acc = accuracy(val_preds, val_targets)
    return(train_acc, val_acc)



def average_perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    (theta_train, theta_0_train) = average_perceptron(train_feature_matrix, train_labels, T)    
    train_preds = classify(train_feature_matrix, theta_train, theta_0_train)
    train_targets = train_labels 
    train_acc = accuracy(train_preds, train_targets)   
    
    val_preds = classify(val_feature_matrix, theta_train, theta_0_train)
    val_targets = val_labels
    val_acc = accuracy(val_preds, val_targets)
    return(train_acc, val_acc)    


def average_passive_aggressive_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L):
    (theta_train, theta_0_train) = average_passive_aggressive(train_feature_matrix, train_labels, T, L)    
    train_preds = classify(train_feature_matrix, theta_train, theta_0_train)
    train_targets = train_labels 
    train_acc = accuracy(train_preds, train_targets)   
    
    val_preds = classify(val_feature_matrix, theta_train, theta_0_train)
    val_targets = val_labels
    val_acc = accuracy(val_preds, val_targets)
    return(train_acc, val_acc)    
    

def extract_words(input_string):
    """
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input
    """
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

def modified_bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input
    """
    dictionary = {} # maps word to unique index
    stop_words = [line.rstrip('\n') for line in open('stopwords.txt')]    
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stop_words:
                dictionary[word] = len(dictionary)
    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix

def extract_additional_features(reviews):
    """
    Inputs a list of string reviews
    Returns a feature matrix of (n,m), where n is the number of reviews
    and m is the total number of additional features
    """
    return np.ndarray((len(reviews), 0))

def extract_final_features(reviews, dictionary):
    """
    Constructs a final feature matrix using the improved bag-of-words and/or additional features
    """
    bow_feature_matrix = extract_bow_feature_vectors(reviews,dictionary)
    additional_feature_matrix = extract_additional_features(reviews)
    return np.hstack((bow_feature_matrix, additional_feature_matrix))

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
