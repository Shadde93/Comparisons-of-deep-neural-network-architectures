%% Multi-Layer Perceptron (MLP)

clear all; close all;
rng('default');

%% load data

data = load('binMNIST.mat');
Xtrain = data.bindata_trn;
Xtest = data.bindata_tst;
ytrain = data.digtargets_trn;
ytest = data.digtargets_tst;


%% set model parameters
model.n_output = 10;
model.n_features = size(Xtrain,2);
model.n_hidden = 10;
model.l1 = 0;
model.l2 = 1;
model.epochs = 100;
model.eta = 0.001;
model.alpha = 0.001;
model.decrease_const = 0.00001;
model.minibatches = 50;

%% MLP
[model] = mlp_fit(Xtrain, ytrain, model);

y_train_pred = mlp_predict(Xtrain, model);
y_test_pred = mlp_predict(Xtest, model);


%% compute accuracy
acc = sum(ytrain' == y_train_pred) / size(Xtrain,1);
fprintf('Training accuracy: %.2f\n',(acc * 100));

acc = sum(ytest' == y_test_pred) / size(Xtest,1);
fprintf('Test accuracy: %.2f\n',(acc * 100));

%% generate plots
figure;
plot(model.cost/size(Xtrain,1)); 
title('MLP Training error'); 
ylabel('RMSE'); xlabel('Epochs x Minibatch');

%%
rmseTr = sum((y_train_pred-ytrain').^2)*1/size(ytrain,1);
rmseTe = sum((y_test_pred-ytest').^2)*1/size(ytest,1);

