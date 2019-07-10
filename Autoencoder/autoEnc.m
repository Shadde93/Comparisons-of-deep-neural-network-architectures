close all

%-----Load data-----
% For autoencoders in MATLAB, the colums are the samples and rows features 
data = load('binMNIST.mat');
bindata_trn = data.bindata_trn';
bindata_tst = data.bindata_tst';
digtargets_trn = data.digtargets_trn';
digtargets_tst = data.digtargets_tst';

c = cell(1,8000);

for i = 1:8000
    c{1,i} = reshape(bindata_trn(:,i),28,28)';
end

ctest = cell(1,2000);

for i = 1:2000
    ctest{1,i} = reshape(bindata_tst(:,i),28,28)';
end

%------Autoencoder-------

hiddenSize = 150;
epochs = 100;
autoenc = trainAutoencoder(c,hiddenSize,'MaxEpochs',epochs,'SparsityRegularization',0); %,'DecoderTransferFunction','purelin');

tester = predict(autoenc,c);

% entry in training target - value of target
% 1 - 6
% 3 - 1 
% 4 - 5
% 5 - 7
% 7 - 4
% 9 - 2
% 10 - 9
% 12 - 0
% 16 - 3
% 32 - 8
showStuff = [12 3 9 16 7 4 1 5 32 10];
figure(1);
for i = 1:length(showStuff)
    subplot(5,2,i)
    imshow(tester{showStuff(i)})
end

if (hiddenSize == 50 || hiddenSize == 100)
    figure;
    plotWeights(autoenc)
end

figure;
tester2 = predict(autoenc,ctest);

show2 = [19 4 8 1 3 2 15 9 7 6];

for i = 1:length(show2)
    subplot(5,2,i)
    imshow(tester2{show2(i)});
end

mse = zeros(1,2000);
for i = 1:2000
    mse(i) = sum(sum((tester2{i} - ctest{i}).^2));
end

disp('mse = ')
sum(mse/(2000*728))

disp('rmse = ')
sqrt(sum(mse/(2000*728)))





% for i = 1:length(show2)
%     subplot(5,2,i)
%     imshow(reshape(bindata_tst(:,show2(i)),28,28)');
% end



