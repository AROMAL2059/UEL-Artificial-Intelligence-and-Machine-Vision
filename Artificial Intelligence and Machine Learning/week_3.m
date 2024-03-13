% Aromal 14/02/2024 
% To create Neural Networks
% clear all space
clear
close all
clc
% load the dataset
 load crab_dataset.mat
 %Create a two-layer feed-forward network. The network has one hidden layer
% with 10 neurons. The training algorithm 'trainlm' is selected.
net = feedforwardnet(10, 'traingdx'); 


%Configure the network inputs and outputs to best match input and target data
net = configure(net, crabInputs, crabTargets); 
%Train the neural network (net) with inputs (crabInputs) and target (crabTargets)
[net,tr] = train(net,crabInputs,crabTargets); 
testInput = crabInputs(:,tr.testInd);
testTarget = crabTargets(:,tr.testInd);
testY = net(testInput); 
plotconfusion(testTarget,testY)