clear;close all; clc;


data = load('binMNIST.mat')


Xtrain = data(1).bindata_trn;
Ytrain = data(1).digtargets_trn;

Xtest = data(1).bindata_tst;
Ytest = data(1).digtargets_tst;

%% RBM training klar kod via matlab 4 olika rbm med olika units
inputdata = Xtrain;
outputdata = Ytrain;
unit = 50;


[n, h] = size(Xtrain);
 
rbm1 = randRBM(h , unit);
rbm1 = pretrainRBM( rbm1, inputdata );

rbm2 = randRBM(h , unit+25);
rbm2 = pretrainRBM( rbm2, inputdata );

rbm3 = randRBM(h , unit*2);
rbm3 = pretrainRBM( rbm3, inputdata );

rbm4 = randRBM(h , unit*3);
rbm4 = pretrainRBM( rbm4, inputdata );


%% rätt indexering 0-9


showStuff = [12 3 9 16 7 4 1 5 32 10];
% Gibbs sampling step 0
vis0 = double(Xtrain(showStuff,:)); % Set values of visible nodes
    
hid0 = v2h( rbm1, vis0 );  % Compute hidden nodes
    
% Gibbs sampling step 1
bhid0 = double( rand(size(hid0)) < hid0 );
   
vis1 = h2v( rbm1, bhid0 );  % Compute visible nodes
    
    %% Plot pictures of trained 0-9
    idx = [12 3 9 18 7 4 1 5 32 28];
    x = Xtrain(idx,:)
    figure(1)
    for i = 1:10
        subplot(2,5,i)
        imshow(reshape(x(i,:),28,28)')
    
    end

    
    %% Plot rmse 
    figure(2)
    hold on
    rmse1 = rbm1.rmse;
    rmse2 = rbm2.rmse;
    rmse3 = rbm3.rmse;
    rmse4 = rbm4.rmse;
    
    plot(1:100,rmse1,1:100,rmse2,1:100,rmse3,'g',1:100,rmse4)
    legend('50 units','75 units','100 units','150 units')
    title('Reconstruction RMS error')
    xlabel('Epochs')
    ylabel('rmse')
    
%% plot f�r vikter W 
%W = TrainedStruct(1).W
W = rbm1(1).W
figure(3)
for i= 1:unit
    subplot(5,10,i)
    imshow(reshape(W(:,i),28,28)')
end

%% Test for all 4 rbms


% Gibbs sampling step 0
vistest1 = double(Xtest); % Set values of visible nodes

hidtest = v2h( rbm1, vistest1 );  % Compute hidden nodes

% Gibbs sampling step 1
bhidtest = double( rand(size(hidtest)) < hidtest );

vistest2 = h2v( rbm1, bhidtest );  % Compute visible nodes

% Gibbs sampling step 0--------------rbm2
visr1 = double(Xtest); % Set values of visible nodes

hidtestr1 = v2h( rbm2, visr1 );  % Compute hidden nodes

% Gibbs sampling step 1
bhidtestr1 = double( rand(size(hidtestr1)) < hidtestr1 );

vistestr12 = h2v( rbm2, bhidtestr1 );  % Compute visible nodes

% Gibbs sampling step 0-----------------rbm3
visr2 = double(Xtest); % Set values of visible nodes

hidtestr2 = v2h( rbm3, visr2 );  % Compute hidden nodes

% Gibbs sampling step 1
bhidtestr2 = double( rand(size(hidtestr2)) < hidtestr2 );

vistestr13 = h2v( rbm3, bhidtestr2 );  % Compute visible nodes

% Gibbs sampling step 0-----------------rbm4
visr3 = double(Xtest); % Set values of visible nodes

hidtestr3 = v2h( rbm4, visr3 );  % Compute hidden nodes

% Gibbs sampling step 1
bhidtestr3 = double( rand(size(hidtestr3)) < hidtestr3 );

vistestr14 = h2v( rbm4, bhidtestr3 );  % Compute visible nodes

%% plot reconstructed test data
showStuffs = [19 4 8 1 3 2 15 9 7 6];
vistests = double(Xtest(showStuffs,:));

hidtests = v2h( rbm1, vistests );  % Compute hidden nodes

% Gibbs sampling step 1
bhidtests = double( rand(size(hidtests)) < hidtests );

vistests2 = h2v( rbm1, bhidtests );  % Compute visible nodes

figure(4)
    for i = 1:10
        subplot(2,5,i)
        imshow(reshape(vistests2(i,:),28,28)')  
    end
    %% rmse for all 4 rbms

err = power(vistest1 - vistest2, 2);
rmse50 = sqrt( sum(err(:)) / numel(err) )

err = power(visr1 - vistestr12 , 2);
rmse75 = sqrt( sum(err(:)) / numel(err) )
    
err = power(visr2 - vistestr13, 2);
rmse100 = sqrt( sum(err(:)) / numel(err) )

err = power(visr3 - vistestr14, 2);
rmse150 = sqrt( sum(err(:)) / numel(err) )







