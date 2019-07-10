clear;close all; clc;


data = load('binMNIST.mat')


Xtrain = data(1).bindata_trn;
Ytrain = data(1).digtargets_trn;

Xtest = data(1).bindata_tst;
Ytest = data(1).digtargets_tst;
%% RBM training klar kod via matlab
inputdata = Xtrain;
outputdata = Ytrain;
unit = 150;

[n, h] = size(Xtrain);
 
rbm = randRBM(h , unit);
rbm = pretrainRBM( rbm, inputdata );


    showStuff = [12 3 9 16 7 4 1 5 32 10];
    % Gibbs sampling step 0
    vis0 = double(Xtrain); % Set values of visible nodes
    
    hid0 = v2h( rbm, vis0 );  % Compute hidden nodes
   
    
    %% Train softmax net
labels = zeros(10,length(Ytrain));
for i = 1 : size(Ytrain)
    labels(Ytrain(i)+1,i) = 1;
end
input = hid0';
net = trainSoftmaxLayer(input,labels)
    
 
%% 

%Y = net(input);
a = net(input);
[m,i] = max(a);
index = i-1;

idx = (index == Ytrain');

trueclass = Ytrain(idx);

correctClass = length(trueclass)/8000
%% Test

vistest1 = double(Xtest); % Set values of visible nodes

hidtest = v2h( rbm, vistest1 );  % Compute hidden nodes

% % Gibbs sampling step 1
% bhidtest = double( rand(size(hidtest)) < hidtest );
% 
% vistest2 = h2v( rbm, bhidtest );  % Compute visible nodes

Y = net(hidtest');

[m,i2] = max(Y);
index2 = i2-1;

indx = (index2 == Ytest');
 
trueclass2 = Ytest(indx);
 
correctClass2 = length(trueclass2)/2000


