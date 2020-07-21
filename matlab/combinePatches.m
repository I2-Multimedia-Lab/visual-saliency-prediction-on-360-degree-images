function sal = combinePatches(inputImage, outputImage)

img = imread(inputImage);
%截取文件名
left = findstr(inputImage,'/'); %取出字符为‘/’的下标
index = findstr(inputImage, '_'); %取出字符为‘_’的下标
str = inputImage(left+1:index-1); %取出字符串从下标left到right的子字符串


tmpFolder = ['test/',str,'/tmp'];
outCNNFolder = tmpFolder;

offsets(1, :) = [90, 0];
offsets(2, :) = [-90, 0];
patchPx = 500;

for xOffset=0:90:270    
    offsets(size(offsets, 1) + 1, :) = [0, xOffset];
end

imgSz = size(img);
load([tmpFolder,'/indices.mat']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CNN-Saliency Patches combination    %  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nbrPatches = 6;
patchPrefix = 'PatchSaliency';

predictionImg = zeros([imgSz(1), imgSz(2)]);
predictionNbr = zeros(size(predictionImg));  
    
for i=1:nbrPatches       
    curPredictionImg = zeros(size(predictionImg));        
    curPredictionNbr = zeros(size(predictionImg));
    
    %Update when reading binary files
    curPatchFile = [outCNNFolder, '/', patchPrefix, '', num2str(i), '.png'];
    
    curPredPatch = imread(curPatchFile);
    curPredPatch = imresize(curPredPatch, [patchPx patchPx]);
    
    curInd = indices(i, :, :);

    %Change using Green as test
    curPredictionImg(curInd) = curPredPatch;        
    curPredictionNbr(curInd) = 1;

    predictionImg = predictionImg + curPredictionImg;
    predictionNbr = predictionNbr + curPredictionNbr;
end

predictionImg = predictionImg./predictionNbr;    
predictionImg = fillBlanks(predictionImg); 

sal = imgaussfilt(predictionImg, 64);
sal = sal - min(sal(:));
sal = sal/sum(sal(:));

x = sal - min(sal(:));
x = uint8(x/max(x(:))*255);

imwrite(x, outputImage)

end