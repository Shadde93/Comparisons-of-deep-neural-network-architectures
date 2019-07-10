clear;close all; clc;


data = load('binMNIST.mat')


Xtrain = data(1).bindata_trn;
Ytrain = data(1).digtargets_trn;

Xtest = data(1).bindata_tst;
Ytest = data(1).digtargets_tst;
%% Train DBN

inputdata = Xtrain;
outputdata = Ytrain;
unit = 150;

[n, h] = size(Xtrain);
%  
% rbm = randRBM(h , unit);<
% rbm = pretrainRBM( rbm, inputdata );

dbn = randDBN([h, unit, unit]);% unitssss
dbn = pretrainDBN( dbn, inputdata );


%% Train softmax net
labels = zeros(10,length(Ytrain));
for i = 1 : size(Ytrain)
    labels(Ytrain(i)+1,i) = 1;
end


input= dbn.X';
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

hidtest = v2h( dbn.rbm{1,1}, vistest1 );  % Compute hidden nodes

% Gibbs sampling step 1
% bhidtest = double( rand(size(hidtest)) < hidtest );
% 
% vistest2 = h2v( rbm, bhidtest );  % Compute visible nodes

hidtest2 = v2h( dbn.rbm{2,1}, hidtest );  % Compute hidden nodes

Y = net(hidtest2');

[m,i2] = max(Y);
index2 = i2-1;

indx = (index2 == Ytest');
 
trueclass2 = Ytest(indx);
 
correctClass2 = length(trueclass2)/2000
 








