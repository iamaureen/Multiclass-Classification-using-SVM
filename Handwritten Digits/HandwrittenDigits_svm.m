clc;
clear all;
close all;

%read data: reference: https://www.mathworks.com/help/matlab/ref/importdata.html
X_train = importdata('X_train.mat');
y_train = importdata('y_train.mat'); 
X_test = importdata('X_test.mat'); 
y_test = importdata('y_test.mat');

%transposing the class label vectors
y_train_transpose = transpose(y_train);
y_test_transpose = transpose(y_test);

%initialization
%number of class-10
%number of test samples-3251
SVMModel = cell(10,1);
label = zeros(10,3251);

%1 in the place of index, other class 0
trainingClassLabelsMatrix = full(ind2vec(y_train_transpose,10));

%train the model one-vs-all
for index=1:10
    SVMModel{index} = fitcsvm(X_train,trainingClassLabelsMatrix(index,:),'KernelFunction','polynomial','PolynomialOrder',2);
end

%predict values
for index=1:10
    label(index,:) = predict(SVMModel{index},X_test);
end

%transform into index
predictedLabel=vec2ind(label);

%calculate accuracy
accuracy = sum(y_test_transpose == predictedLabel)/length(y_test_transpose);
accuracyPercentage = 100*accuracy;
fprintf('Accuracy = %f%%\n',accuracyPercentage)

