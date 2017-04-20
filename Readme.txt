Implemented Multiclass Classifier using Support Vector Machine with the following datasets:

Human Activity Datasets
-----------------------
Number of classes: 6
Number of training data: 7352
Number of features: 561
Number of test data: 2947


VIdTIMIT Datasets
-----------------------
Number of classes: 25
Number of training data: 3500
Number of features: 100
Number of test data: 1000

Handwritten Digits Datasets
-----------------------
Number of classes: 10
Number of training data: 500
Number of features: 64
Number of test data: 3251

SVM is trained for each class, and for predicting a test sample, the maximum value returned by all the
SVM are used to decide the final class.