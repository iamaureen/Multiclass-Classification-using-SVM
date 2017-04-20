clc;
clear all;
close all;

X_train = importdata('X_train.mat'); 
y_train = importdata('y_train.mat'); 
X_test = importdata('X_test.mat'); 
y_test = importdata('y_test.mat'); 

%initialization
%number of class-25
%number of test samples-1000
SVMModel = cell(25,1);
label = zeros(25,1000);

%1 in the place of index, other class 0
trainingClassLabelsMatrix = full(ind2vec(y_train,25));

%train the model one-vs-all: reference: https://www.mathworks.com/help/stats/fitcsvm.html?s_tid=gn_loc_drop
for index=1:25
    SVMModel{index} = fitcsvm(X_train,trainingClassLabelsMatrix(index,:),'KernelFunction','polynomial','PolynomialOrder',2);
end

%predict values
for index=1:25
    label(index,:) = predict(SVMModel{index},X_test);
end

%transform into index
predictedLabel=vec2ind(label);

%calculate accuracy
accuracy = sum(y_test == predictedLabel)/length(y_test);
accuracyPercentage = 100*accuracy;
fprintf('Accuracy = %f%%\n',accuracyPercentage)
