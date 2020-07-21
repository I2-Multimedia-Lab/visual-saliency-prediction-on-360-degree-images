function success = randomcreatePatches(imgName)

%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Patches creation    %  
%%%%%%%%%%%%%%%%%%%%%%%%%%

img = imread(imgName);

tmpFolder = '/home/cxl/Documents/dataset/ICME2018/test_fixations';
%sph = '/home/cxl/Documents/dataset/ICME2018/test_sph'
patchPx = 500;

success = 0;

%获取imgName的名称　eg: P1, P2
left = findstr(imgName,'P'); %取出字符为‘/’的下标
index = findstr(imgName, '.'); %取出字符为‘_’的下标
str = imgName(left:index-1); %取出字符串从下标left到right的子字符串

for i=-90:10:90
    for j=0:30:360
        [imgPatch, ind, phi, theta] = extractPatch(img, i, j, patchPx, patchPx); %（图片，经度，维度，边长，边长 ）       
        
        curOutFile = [str, 'fix_Patch', num2str(i), num2str(j), '.png'];        
        imwrite(imgPatch, [tmpFolder, '/', curOutFile]);
        
%         sphCoordFile = [str, '_PatchSC', num2str(i), num2str(j), '.bin'];
%         fileID = fopen([sph, '/', sphCoordFile], 'w');
%         fwrite(fileID, patchPx, 'uint16');
%         fwrite(fileID, phi, 'single');
%         fwrite(fileID, theta, 'single');
%         fclose(fileID);
        
    end
end

success = 1;

end
