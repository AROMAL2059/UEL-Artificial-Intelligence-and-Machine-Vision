% Create Transfer Learning based Network for Classification in MATLAB using GoogleNet

close all

clear

clc

%Load Data

imds = imageDatastore('MerchData', ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');

%

labelCount = countEachLabel(imds);

img = readimage(imds,1);

size(img);

%Training and Validation data Sets

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);

idx = randperm(numTrainImages,16);

figure

for i = 1:16

subplot(4,4,i)

I = readimage(imdsTrain,idx(i));

imshow(I)

end
%pretrained Data

net = googlenet;

% Create a layer graph from the network
lgraph = layerGraph(net);

% Find layers that can be replaced
[learnableLayer, classLayer] = findLayersToReplace(lgraph);

% Display the layers that can be replaced
learnableLayer
classLayer

% Analyze and visualize the network
analyzeNetwork(net);

inputSize = net.Layers(1).InputSize

layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1, numClasses, ...
        'Name', 'new_conv', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);
end
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');

lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);

plot(lgraph)

ylim([0,10])
