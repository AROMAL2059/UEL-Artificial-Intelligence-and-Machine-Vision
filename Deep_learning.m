% Aromal 21/02/2024
% This script implements a deep neural network architecture in MATLAB and trains and tests it using the Digit dataset

% Close all open figures
close all;

% Clear the workspace
clear;

% Clear the command window
clc;

% Define the path to a directory containing a dataset of digit images
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', 'nndemos', 'nndatasets', 'DigitDataset');

% Create an ImageDatastore object using the imageDatastore function
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Display a random selection of 20 images from the dataset
figure;
perm = randperm(10000,20);
for i = 1:20 
    subplot(4,5,i); 
    imshow(imds.Files{perm(i)}); 
end

% Count the number of images for each label in the dataset
labelCount = countEachLabel(imds);

% Read the first image in the dataset and display its size
img = readimage(imds,1); 
size(img);

% Split the dataset into training and validation sets
numTrainFiles = 750; 
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

% Define the layers of the neural network
layers = [ 
    imageInputLayer([28 28 1]) 
    convolution2dLayer(3,8,'Padding','same') 
    batchNormalizationLayer 
    reluLayer 
    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,16,'Padding','same') 
    batchNormalizationLayer 
    reluLayer 
    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,32,'Padding','same') 
    batchNormalizationLayer 
    reluLayer 
    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,64,'Padding','same') 
    batchNormalizationLayer 
    reluLayer
    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,64,'Padding','same') 
    batchNormalizationLayer 
    reluLayer
    fullyConnectedLayer(10) 
    softmaxLayer 
    classificationLayer
]; 

% Define the training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network
net = trainNetwork(imdsTrain,layers,options);

% Classify the validation set
YPred = classify(net,imdsValidation); 

% Get the labels of the validation set
YValidation = imdsValidation.Labels; 

% Calculate the accuracy of the network
accuracy = sum(YPred == YValidation)/numel(YValidation);
accuracy
