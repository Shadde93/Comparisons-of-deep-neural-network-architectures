close all

%-----Load data-----
% For autoencoders in MATLAB, the colums are the samples and rows features 
data = load('binMNIST.mat');
bindata_trn = data.bindata_trn';
bindata_tst = data.bindata_tst';
digtargets_trn = data.digtargets_trn';
digtargets_tst = data.digtargets_tst';

sz = size(bindata_trn,2);
labels = zeros(10,sz);
labels2 = zeros(10,2000);

for i = 1:sz
    labels(digtargets_trn(i)+1,i) = 1;
end

for i = 1:2000
    labels2(digtargets_tst(i)+1,i) = 1;
end


%----Stacking----
layers = 3;
hiddenSize1 = 150;
hiddenSize2 = 150;
hiddenSize3 = 150;
epochs = 50;

autoenc1 = trainAutoencoder(bindata_trn,hiddenSize1,'MaxEpochs',epochs,'SparsityRegularization',0.01);

feat1 = encode(autoenc1, bindata_trn);

if layers == 1
    softnet = trainSoftmaxLayer(feat1,labels,'MaxEpochs',400);
    deepnet = stack(autoenc1,softnet);
end

if layers == 2
    autoenc2 = trainAutoencoder(feat1,hiddenSize2,'MaxEpochs',epochs,'SparsityRegularization',0.01);
    feat2 = encode(autoenc2,feat1);
    softnet = trainSoftmaxLayer(feat2,labels,'MaxEpochs',400);
    deepnet = stack(autoenc1,autoenc2,softnet);
end

if layers == 3
    autoenc2 = trainAutoencoder(feat1,hiddenSize2,'MaxEpochs',epochs,'SparsityRegularization',0.01);
    feat2 = encode(autoenc2,feat1);
    autoenc3 = trainAutoencoder(feat2,hiddenSize3,'MaxEpochs',epochs,'SparsityRegularization',0.01);
    feat3 = encode(autoenc3,feat2);
    softnet = trainSoftmaxLayer(feat3,labels,'MaxEpochs',400);
    deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);
end

[M,I] = max(deepnet(bindata_trn));

correct = sum(I-1 == digtargets_trn)/length(digtargets_trn);

I(1:10)-1
digtargets_trn(1:10)

[~,I2] = max(deepnet(bindata_tst));

correct2 = sum(I2-1 == digtargets_tst)/length(digtargets_tst);



% [~,Itst] = max(deepnet(bindata_tst));
% 
% sum
%-----MLP-------

% x = feat1;
% t = digtargets_trn;
% 
% % Choose a Training Function
% % For a list of all training functions type: help nntrain
% % 'trainlm' is usually fastest.
% % 'trainbr' takes longer but may be better for challenging problems.
% % 'trainscg' uses less memory. Suitable in low memory situations.
% trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
% 
% % Create a Fitting Network
% hiddenLayers = [20]; % [#hidden nodes in first layer, #second layer and so on...]
% net = fitnet(hiddenLayers,trainFcn); %  
% 
% % Setup Division of Data for Training, Validation, Testing
% net.divideParam.trainRatio = 50/100;
% net.divideParam.valRatio = 30/100;
% net.divideParam.testRatio = 20/100;
% 
% % Train the Network
% %net.divideFcn = '';
% %net.trainParam.epochs = 300;
% %net.trainParam.goal = 1e-5;
% net.performParam.regularization = 0.01;
% [net,tr] = train(net,x,t);
% 
% % Test the Network
% y = net(x);
% e = gsubtract(t,y);
% 
% performance = perform(net,t,y)
% 
% % View the Network
% %view(net)
% 
% % Plots
% % Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% %figure, plottrainstate(tr)
% %figure, ploterrhist(e)
% %figure, plotregression(t,y)
% %figure, plotfit(net,x,t)
% 
% deepnet = stack(autoenc1,autoenc2,net);
% 
% 
% % Perform fine tuning
% deepnet1 = train(deepnet,bindata_trn(:,1),digtargets_trn(1));
