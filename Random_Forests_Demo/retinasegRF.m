dataDir = 'datasets/training/images/';
maskDir = 'datasets/training/1st_manual/';
validDir = 'datasets/training/mask/';
nSamplesPerImage = 2000;
trainIdx = 21;
testIdx = 21;

scales = [3, 5, 7, 9, 11];

%% Training
img = imread([dataDir, num2str(trainIdx), '_training.tif']);
msk = (imread([maskDir, num2str(trainIdx), '_manual1.gif'])>0);
valid = (imread([validDir, num2str(trainIdx), '_training_mask.gif'])>0);

imgR = double(img(:,:,1));
imgG = double(img(:,:,2));
imgB = double(img(:,:,3));

%% Multiscale feature extraction
featVec = double(zeros([sum(valid(:)), 1]));
for scaleIdx = 1:length(scales)
    kernel = fspecial('gaussian',[3*scales(scaleIdx), 3*scales(scaleIdx)], scales(scaleIdx));
    featR = filter2(kernel, imgR, 'same');
    featVec = cat(2,featVec, featR(valid));
    featG = filter2(kernel, imgR, 'same');
    featVec = cat(2,featVec, featG(valid));
    featB = filter2(kernel, imgR, 'same');
    featVec = cat(2,featVec, featB(valid));
end

%% Arranging a feature vector
featVec = featVec(:,2:end);
labelsVec = msk(valid);

%% Random forest learning
rfObj = TreeBagger(5, featVec, labelsVec, 'Nprint', 1, 'MinLeaf', 500);

%% Testing
img = imread([dataDir, num2str(testIdx), '_training.tif']);
msk = (imread([maskDir, num2str(testIdx), '_manual1.gif'])>0);
valid = (imread([validDir, num2str(testIdx), '_training_mask.gif'])>0);
sz = size(msk);
imgR = double(img(:,:,1));
imgG = double(img(:,:,2));
imgB = double(img(:,:,3));

% Multi-scale feature extraction
featVec = double(zeros([sz(1)*sz(2), 1]));
for scaleIdx = 1:length(scales)
    kernel = fspecial('gaussian',[3*scales(scaleIdx), 3*scales(scaleIdx)], scales(scaleIdx));
    featR = filter2(kernel, imgR, 'same');
    featVec = cat(2,featVec, featR(:));
    featG = filter2(kernel, imgR, 'same');
    featVec = cat(2,featVec, featG(:));
    featB = filter2(kernel, imgR, 'same');
    featVec = cat(2,featVec, featB(:));
end
% Arranging a feature vector
featVec = featVec(:,2:end);
% Prediction
[~,preds] = predict(rfObj, featVec);
labelMask = reshape(preds(:,2), sz(1), sz(2)).*valid;
% Display prediction
figure;
subplot(1,2,1);
imshow(msk)
title('Ground truth mask')
subplot(1,2,2);
imshow(labelMask)
title('Prediction')