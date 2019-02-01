function Labeling_X1()

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

options.WindowStyle = 'normal';
options.Interpreter = 'none';
options.Resize = 'on';
variableNames = {'filename','Class'};
inputDims = [1 10];
inputTitle = 'Class'; 


% Image filename acquisition 
myFolder = uigetdir;
filePattern = fullfile(myFolder, '*.JPG');
jpegFiles = dir(filePattern);
x = {jpegFiles.name};

y=zeros(length(jpegFiles),202500);
y = uint8(y);


% loop thorugh jpg files
for k = 1:length(jpegFiles)
    
  baseFileName = jpegFiles(k).name;
  
  % Image acquisition
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', baseFileName);
  imageArray = imread(fullFileName);
  
  
  % Image show
  figure(1)
  imshow(imageArray);
  title(baseFileName);
  
  % Image class 1 for waste 0 for no waste
  v = inputdlg({'Class'},inputTitle,inputDims,{'0'},options);
  x(2,k) = v;
  
  imageArray = imresize(imageArray, [225 300]);
  imarray = reshape(imageArray,[1,202500]);
  y(k,:)=imarray;
  

  close

end
% Save all file name and classes
x = cell2table(x','VariableNames',variableNames);
writetable(x,fullfile(myFolder, 'label.csv'));

csvwrite(fullfile(myFolder, 'data.csv'),y)

% m = load('demo/data.csv');
% for i = 1:10
%     v = m(i,:,:);
%     v= reshape(v,[225,300,3]);
%     figure(2)
%     imshow(uint8(v))
%     pause(0.5)
% end