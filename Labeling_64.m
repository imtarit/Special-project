numSegments = 16; %must be square of segment along each exis

defs = cell(1,numSegments);
defs(1,:)={'0'};
options.WindowStyle = 'normal';
variableNames = {'filename','image'};
inputname = {};
inputDims = [1 10];
inputTitle = 'Classes';
for i = 1:(numSegments)
    varName = strcat('Segment',num2str(i));
    variableNames(i+2) = {varName};
    inputname(i) = {varName};
end


myFolder = 'Demo';
filePattern = fullfile(myFolder, '*.jpg');
jpegFiles = dir(filePattern);
label = zeros(length(jpegFiles),65);
x = {jpegFiles.name};
% x = cell2table(x','VariableNames',variableNames);

% x = x;


errFiles = {};

for k = 1:length(jpegFiles)
    
  baseFileName = jpegFiles(k).name;
  
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  imageArray = imread(fullFileName);
  
  imlength = size(imageArray,1);
  imWidth = size(imageArray,2);
  figure(1)
  imshow(imageArray);
  v = inputdlg({'Image 0/1'},inputTitle,inputDims);
  x(2,k) = v;
  
  if strcmp(v,'0')
      v = defs;
  elseif strcmp(v,'1')
      segementAxis = sqrt(numSegments);
      segmentLength = uint64(floor(imlength/segementAxis));
      segmentWidth = uint64(floor(imWidth/segementAxis));
      close
      K = 1;
      for i = 1:segementAxis
          for j = 1:segementAxis
              stX = (i-1)*segmentLength+1;
              stY = (j-1)*segmentWidth+1;
              enX = i*segmentLength;
              enY = j*segmentWidth;
              imSegment = imageArray(stX:enX,stY:enY,:);
              figure(2)
              subplot(segementAxis,segementAxis,K)
              imshow(imSegment);
              title(inputname(K));
              K = K+1;
          end
      end
      v = inputdlg(inputname,inputTitle,inputDims,defs,options);
  else
      waitfor(msgbox('The class should be 0/1', 'Warning'));
      x(2,k) = {'0'};
      v = defs;
      errFiles(end+1) = {baseFileName};
  end
  
  for i = 1:length(v)
      x(i+2,k) = v(i);
  end
  close




end
x = cell2table(x','VariableNames',variableNames);
writetable(x,'filelist.csv');
% x = cell2table(errFiles);
% writetable(x,'error_file_list.csv');

% xlswrite('error_file_list.csv',errFiles);


filePh = fopen('error_file_list.txt','w');
fprintf(filePh,'%s\n',errFiles{:});
fclose(filePh);
