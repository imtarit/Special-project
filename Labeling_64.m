% function Labeling_64(segementAxis)

%Labeling_64 Divides images into segemnts for waste class labeling 
%
%   segementAxis = Number of segments along x and y axis of the image.
%
%   Show the image to select class between 0 and 1. Class 0 indicates no
%   waste in the bin and class 1 indicates the presence of waste. If class
%   0 is selected then all the segemnts are selcted as class 0 of no waste.
%   If any class 1 is selected then the image is splited into same number 
%   of segemnts along x and y axis equal to segementAxis. There fore total
%   number of segments is equal to segementAxis^2. The user can select the
%   class in a popup dialouge box. If the user give wrong input rather than
%   0 or 1 a popup warning will be showed up, all the classes will be set 
%   to 0, and the file name will be added to error list. In the end all the
%   labels and the file name will be saved into filelist.csv and the list
%   of error files will be saved in error_file_list.txt. 
%   
% (C) Fayeem Aziz, University of Newcastle, Australia



if nargin == 0
    segementAxis = 4; %must be square of segments along each exis
end
segementAxis = 4;
numSegments = segementAxis^2;

% Variable initialisation for input dialouge.
defs = cell(1,numSegments); % default class = 0
defs(1,:)={'0'};
options.WindowStyle = 'normal';
options.Interpreter = 'none';
options.Resize = 'on';
variableNames = {'filename','image'};
inputname = {};
inputDims = [1 10];
inputTitle = 'Classes'; 
for i = 1:(numSegments)
    varName = strcat('Segment',num2str(i)); %Segment names
    variableNames(i+2) = {varName};
    inputname(i) = {varName};
end

% Image filename acquisition 
myFolder = uigetdir;
filePattern = fullfile(myFolder, '*.JPG');
jpegFiles = dir(filePattern);
x = {jpegFiles.name};


errFiles = {}; % List of error files

% loop thorugh jpg files
for k = 1:length(jpegFiles)
    
  baseFileName = jpegFiles(k).name;
  
  % Image acquisition
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', baseFileName);
  imageArray = imread(fullFileName);
  
  % Image show
  imlength = size(imageArray,1);
  imWidth = size(imageArray,2);
  figure(1)
  imshow(imageArray);
  
  % Image class 1 for waste 0 for no waste
  v = inputdlg({'Image 0/1'},inputTitle,inputDims,{'0'},options);
  x(2,k) = v;
  
  if strcmp(v,'0') % If no waste all segements are 0 class 
      v = defs;
  elseif strcmp(v,'1') % If waste then segments
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
              % SHow segemtns
              figure(2)
              subplot(segementAxis,segementAxis,K)
              imshow(imSegment);
              title(inputname(K));
              K = K+1;
          end
      end
      
      % Segment classes
      v = inputdlgcol(inputname,inputTitle,inputDims,defs,options,2);
  else % If worng input then all classes are 0 add to error list
      waitfor(msgbox('The class should be 0/1', 'Warning'));
      x(2,k) = {'0'};
      v = defs;
      errFiles(end+1) = {baseFileName};
  end
  
  % Set classes to table x
  for i = 1:length(v)
      x(i+2,k) = v(i);
  end
  close




end
% Save all file name and classes
x = cell2table(x','VariableNames',variableNames);
writetable(x,fullfile(myFolder, 'filelist.csv'));

% Save list of error files
filePh = fopen(fullfile(myFolder, 'error_file_list.txt'),'w');
fprintf(filePh,'%s\n',errFiles{:});
fclose(filePh);
