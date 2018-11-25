import pandas

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier

def read_CSV(name):
    ds = pandas.read_csv(name)
    return ds

def describe_data_set(data_set, column):
    print(data_set[column].value_counts())
    print('\n')

def normalize_columns(columns, data_set):

    for column in columns:
        normalized_df = (data_set[column] - data_set[column].min()) / (data_set[column].max() - data_set[column].min())
        data_set[column] = normalized_df

def drop_features(columns, data_set):

    for column in columns:
        data_set.drop(column, 1)

    return data_set

def drop_columns(to_drop, x):

    for column in to_drop:
        x = x.drop(column, 1)

def show_results(method, validation, accuracy, conf_matrix = 0, mae = '-', mse = '-', rmse = '-'):

    print(method + ' cross validation score : ' + (str)(validation.mean()))
    print(method + ' accuracy : ' + (str)(accuracy))

    if(mae != '-'):
        print('Mean Absolute Error : ' + (str)(mae))

    if(mse != '-'):
        print('Mean Squared Error : ' + (str)(mse))

    if(rmse != '-'):
        print('Root Mean Squared Error : ' + (str)(rmse))

    if(conf_matrix != 0):
        print('Confusion matrix : ')
        print(conf_matrix)

    print('\n')

def show_results_logistic_regression(method, validation, accuracy, conf_matrix):

    print(method + ' cross validation score : ' + (str)(validation.mean()))
    print(method + ' accuracy : ' + (str)(accuracy))
    print('Confusion matrix :')
    print(conf_matrix)
    print('\n')

def show_coefficients(x_train, lin_reg):

    coeff = pandas.DataFrame(list(zip(x_train.columns, lin_reg.coef_)), columns = ['features', 'estimatedCoefficients'])
    print(coeff)

def plot_correlations(data_set, feature, label, csv_name):

    pandas.crosstab(data_set[feature], data_set[label]).plot(kind='bar')
    plt.savefig(csv_name)

def plot_error(x_test, y_test, lin_reg):

    prediction = lin_reg.predict(x_test)

    plt.scatter(prediction, prediction - y_test, c = 'b', s = 40, alpha = 0.5)
    plt.hlines(y = 0, xmin = 0, xmax = 25)
    plt.title('The error on the test data set')

    plt.show()

def plot_estimators_error(x_train, y_train, x_test, y_test):

    accuracies = np.array(range(100))
    accuracies = accuracies.astype(np.float32)

    for i in range(100):
        rand_for = RandomForestClassifier(n_estimators=i + 1, criterion='gini')
        rand_for.fit(x_train, y_train)
        prediction = rand_for.predict(x_test);
        accuracy = accuracy_score(prediction, y_test)
        accuracies[i] = accuracies[i].astype(np.float32)
        accuracies[i] = accuracy

    print(accuracies)
    plt.plot(range(100), accuracies)
    plt.xlabel('No of estimators')
    plt.ylabel('Model error')
    plt.title('The plot of the error while increasing the estimators')
    plt.show()

def logistic_regression(x_train, y_train, x_test, y_test):

    log_reg = LogisticRegression(multi_class='multinomial', solver='saga', penalty='l1', tol=0.01, random_state=1856)
    validation = cross_val_score(log_reg, x_train, y_train, cv=5)

    log_reg.fit(x_train, y_train)

    prediction = log_reg.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    conf_matrix = confusion_matrix(y_test, prediction)
    show_results_logistic_regression('Logistic regression', validation, accuracy, conf_matrix)

def linear_regression(x_train, y_train, x_test, y_test):

    lin_reg = Ridge(solver = 'sag', alpha = 0.1)
    validation = cross_val_score(lin_reg, x_train, y_train, cv=5)

    lin_reg.fit(x_train, y_train)

    plot_error(x_test, y_test, lin_reg)

    prediction = lin_reg.predict(x_test)
    accuracy = r2_score(y_test, prediction)
    mae  = mean_absolute_error(y_test, prediction)
    mse  = mean_squared_error(y_test, prediction)
    rmse = mean_squared_log_error(y_test, prediction)

    #show_coefficients(x_train, lin_reg)
    show_results('Linear regression', validation, accuracy, 0, mae, mse, rmse)

def knn_classifier(x_train, y_train, x_test, y_test):

    knn_class = KNeighborsClassifier()
    validation = cross_val_score(knn_class, x_train, y_train, cv=5)

    knn_class.fit(x_train, y_train)

    prediction = knn_class.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    conf_matrix = confusion_matrix(y_test, prediction)

    show_results_logistic_regression('KNN', validation, accuracy, conf_matrix)

def knn_regressor(x_train, y_train, x_test, y_test):

    knn_reg = KNeighborsRegressor()
    validation = cross_val_score(knn_reg, x_train, y_train, cv=5)

    knn_reg.fit(x_train, y_train)

    plot_error(x_test, y_test, knn_reg)

    prediction = knn_reg.predict(x_test)
    accuracy = r2_score(y_test, prediction)
    mae  = mean_absolute_error(y_test, prediction)
    mse  = mean_squared_error(y_test, prediction)
    rmse = mean_squared_log_error(y_test, prediction)

    show_results('KNN', validation, accuracy, 0, mae, mse, rmse)

def random_forest_classifier(x_train, y_train, x_test, y_test):

    rand_for = RandomForestClassifier(n_estimators=20, criterion='gini', oob_score=True)

    rand_for.fit(x_train, y_train)

    prediction = rand_for.predict(x_test)
    validation = rand_for.oob_score_
    accuracy = accuracy_score(prediction, y_test)
    conf_matrix = confusion_matrix(y_test, prediction)

    show_results_logistic_regression('Random forests', validation, accuracy, conf_matrix)
    print(classification_report(prediction, y_test))

#Classification - given a few features of each dog,predict the breed
def task_1(csv_name):

    #reading the data set
    data_set = read_CSV(csv_name)

    #describe the data set
    #describe_data_set(data_set, 'Breed Name')

    #get data set size
    n = len(data_set)

    #normalize the data set
    normalize_columns(['Weight(g)', 'Height(cm)'], data_set)

    #build the features and the labels
    y = data_set['Breed Name']
    x = data_set.drop('Breed Name', 1)

    #NaN -> mean
    x.fillna(x.mean(), inplace=True);

    #one-hot encode some features
    one_hot_energy_level = pandas.get_dummies(data_set['Energy level'], prefix='energy')
    one_hot_attention_needs = pandas.get_dummies(data_set['Attention Needs'], prefix='attention')
    one_hot_coat_length = pandas.get_dummies(data_set['Coat Lenght'], prefix='coat')
    one_hot_sex = pandas.get_dummies(data_set['Sex'], prefix='sex')

    #remove useless features
    dropped_columns = ['Longevity(yrs)', 'Sex', 'Owner Name', 'Energy level', 'Attention Needs', 'Coat Lenght']
    for column in dropped_columns :
        x = x.drop(column, 1)

    #adding the encoded features
    #x = pandas.concat([x, one_hot_energy_level], axis=1)
    #x = pandas.concat([x, one_hot_attention_needs], axis=1)
    #x = pandas.concat([x, one_hot_coat_length], axis=1)
    #x = pandas.concat([x, one_hot_sex], axis=1)

    #split the data set in 3/4 training data and 1/4 test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = 5)

    #Logistic Regression
    logistic_regression(x_train, y_train, x_test, y_test)

    #KNN Classification
    knn_classifier(x_train, y_train, x_test, y_test)

    #Random forest
    random_forest_classifier(x_train, y_train, x_test, y_test)

#Regression - given a few features of each dog,predict the longevity
def task_2(csv_name):

    # reading the data set
    data_set = read_CSV(csv_name)

    # get data set size
    n = len(data_set)

    #normalize the data set
    normalize_columns(['Weight(g)', 'Height(cm)'], data_set)

    #build the features and the labels
    y = data_set['Longevity(yrs)']
    x = data_set.drop('Longevity(yrs)', 1)

    #NaN -> mean
    x.fillna(x.mean(), inplace=True);

    #one-hot encode some features
    one_hot_energy_level = pandas.get_dummies(data_set['Energy level'], prefix='energy')
    one_hot_attention_needs = pandas.get_dummies(data_set['Attention Needs'], prefix='attention')
    one_hot_coat_length = pandas.get_dummies(data_set['Coat Lenght'], prefix='coat')
    one_hot_sex = pandas.get_dummies(data_set['Sex'], prefix='sex')

    #remove useless features
    dropped_columns = ['Breed Name', 'Sex', 'Owner Name', 'Energy level', 'Attention Needs', 'Coat Lenght']
    for column in dropped_columns :
        x = x.drop(column, 1)

    #adding the encoded features
    #x = pandas.concat([x, one_hot_energy_level], axis=1)
    #x = pandas.concat([x, one_hot_attention_needs], axis=1)
    #x = pandas.concat([x, one_hot_coat_length], axis=1)
    #x = pandas.concat([x, one_hot_sex], axis=1)

    #split the data set in 3/4 training data and 1/4 test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 5)

    #Linear Regression
    linear_regression(x_train, y_train, x_test, y_test)

    #KNN Regression
    knn_regressor(x_train, y_train, x_test, y_test)
