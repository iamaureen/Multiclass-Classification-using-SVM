clc;
clear all;
close all;

%read data: reference: https://www.mathworks.com/help/matlab/ref/importdata.html
X_train = importdata('X_train.txt');
y_train = importdata('y_train.txt'); 
X_test = importdata('X_test.txt'); 
y_test = importdata('y_test.txt');

%transposing the class label vectors
y_train_transpose = transpose(y_train);
y_test_transpose = transpose(y_test);

%initialization
%number of class-6
%number of test samples-2947
SVMModel = cell(6,1);
label = zeros(6,2947);

%1 in the place of index, other class 0
trainingClassLabelsMatrix = full(ind2vec(y_train_transpose,6));

%train the model one-vs-all
for index=1:6
    SVMModel{index} = fitcsvm(X_train,trainingClassLabelsMatrix(index,:),'KernelFunction','polynomial','PolynomialOrder',2);
end

%predict values
for index=1:6
    label(index,:) = predict(SVMModel{index},X_test);
end

%transform into index
predictedLabel=vec2ind(label);

%calculate accuracy
accuracy = sum(y_test_transpose == predictedLabel)/length(y_test_transpose);
accuracyPercentage = 100*accuracy;
fprintf('Accuracy = %f%%\n',accuracyPercentage)

