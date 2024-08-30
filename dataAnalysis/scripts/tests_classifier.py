import time
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import train_test_split



def prediction_test(classifier_model, X_test, y_test, return_time = False):
    # Predykcja
    start_pred = time.time()
    y_pred = classifier_model.predict(X_test)
    t_pred = time.time() - start_pred
    
    # Jakość klasyfikacji
    acc = metrics.accuracy_score(y_test, y_pred)

    # Macierz pomyłek
    cm = metrics.confusion_matrix(y_test, y_pred, labels=classifier_model.classes_, normalize='true')

    if not return_time:
        return acc, cm

    return acc, cm, t_pred


def classifier_test(classifier_model, X_train, X_test, y_train, y_test, return_time = False):
    # Uczenie modelu
    start_learn = time.time()
    classifier_model.fit(X_train, y_train)
    t_learn = time.time() - start_learn
    
    acc, cm, t_pred = prediction_test(classifier_model, X_test, y_test, return_time)


    if not return_time:
        return acc, cm

    return acc, cm, t_learn, t_pred


def complex_classifier_test(data, target, n_classes, classifiers, classifiers_docstr, tests_size, seeds):
    
    n_iter = len(seeds)
    
    # Memory allocation
    accuracy_matrix  = np.zeros( ( len(tests_size), len(classifiers) ) )
    confusion_matrix = np.zeros( ( len(tests_size), len(classifiers), n_classes, n_classes ) )
    training_time_matrix   = np.zeros( ( len(tests_size), len(classifiers) ) )
    prediction_time_matrix = np.zeros( ( len(tests_size), len(classifiers) ) )

    acc_train_matrix  = np.zeros( ( len(tests_size), len(classifiers) ) )
    conf_train_matrix = np.zeros( ( len(tests_size), len(classifiers), n_classes, n_classes ) )

    # --- CLASSIFIER TESTS ---

    # For each classifiers 
    print("Test klasyfikatorów: ")
    p_bar_class = tqdm(classifiers)
    for j, classifier in enumerate(p_bar_class):

        # For each test size
        
        for i, t_size in enumerate(tests_size):
            p_bar_class.set_description('{: <50}'.format('{}, Test Size: {}'.format(classifiers_docstr[j], t_size)))


            # For each iteration
            for seed in seeds:
                # Shuffle and roll test size
                X_train, X_test, y_train, y_test = train_test_split(data, target,           
                                                                    test_size=t_size, random_state=seed)

                # Classifier test 
                accuracy, confusion, t_learn, t_pred = classifier_test(classifier, X_train, X_test, y_train, y_test, True)

                # Prediction training set
                acc_train, conf_train, _ = prediction_test(classifier, X_train, y_train, True)

                # Adding evaluation statistics
                accuracy_matrix[i,j]        += accuracy
                confusion_matrix[i,j,:,:]   += confusion
                training_time_matrix[i,j]   += t_learn
                prediction_time_matrix[i,j] += t_pred

                acc_train_matrix[i,j]        += acc_train
                conf_train_matrix[i,j,:,:]   += conf_train


    p_bar_class.set_description('Done')

    # Normalize evaluation statistics
    accuracy_matrix         /= n_iter
    training_time_matrix    /= n_iter
    prediction_time_matrix  /= n_iter

    acc_train_matrix        /= n_iter
    conf_train_matrix       /= n_iter

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            confusion_matrix[i,j,:,:] = confusion_matrix[i,j,:,:] / n_iter   

    return accuracy_matrix, confusion_matrix, acc_train_matrix, conf_train_matrix , training_time_matrix, prediction_time_matrix 