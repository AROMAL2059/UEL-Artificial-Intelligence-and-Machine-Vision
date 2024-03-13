
close all

clear

clc
unzip('MerchData.zip'); 
imds = imageDatastore('MerchData', ...
 'IncludeSubfolders',true, ...
 'LabelSource','foldernames'); 
% Count the number of images for each label in the dataset
labelCount = countEachLabel(imds);

% Read the first image in the dataset and display its size
img = readimage(imds,1); 
size(img);
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized'); 
numTrainImages = numel(imdsTrain.Labels); 
idx = randperm(numTrainImages,16); 
figure 
for i = 1:16 
 subplot(4,4,i) 
 I = readimage(imdsTrain,idx(i)); 
 imshow(I) 
end