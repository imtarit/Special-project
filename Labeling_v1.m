function varargout = Labeling_v1(varargin)
% LABELING_V1 MATLAB code for Labeling_v1.fig
%      LABELING_V1, by itself, creates a new LABELING_V1 or raises the existing
%      singleton*.
%
%      H = LABELING_V1 returns the handle to a new LABELING_V1 or the handle to
%      the existing singleton*.
%
%      LABELING_V1('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LABELING_V1.M with the given input arguments.
%
%      LABELING_V1('Property','Value',...) creates a new LABELING_V1 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Labeling_v1_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are xpassed to Labeling_v1_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Labeling_v1

% Last Modified by GUIDE v2.5 15-Mar-2019 16:03:52

%Fayeem Aziz February 2019

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Labeling_v1_OpeningFcn, ...
                   'gui_OutputFcn',  @Labeling_v1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end


% End initialization code - DO NOT EDIT


% --- Executes just before Labeling_v1 is made visible.
function Labeling_v1_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Labeling_v1 (see VARARGIN)

% Choose default command line output for Labeling_v1
handles.output = hObject;

% Update handles structure

% handles.x = {};

guidata(hObject, handles);

set(handles.startButton,'Enable','off')
% set(handles.uibuttongroup2,'selectedobject',handles.class_0)




% UIWAIT makes Labeling_v1 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Labeling_v1_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in Class_0.
% function Class_0_Callback(hObject, eventdata, handles)
% hObject    handle to Class_0 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Class_0


% --- Executes on button press in Class_1.
% function Class_1_Callback(hObject, eventdata, handles)
% hObject    handle to Class_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Class_1


% --- Executes on button press in Class_2.
% function Class_2_Callback(hObject, eventdata, handles)
% hObject    handle to Class_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Class_2


% --- Executes on button press in Class_3.
% function Class_3_Callback(hObject, eventdata, handles)
% hObject    handle to Class_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Class_3


% --- Executes on button press in Class_4.
% function Class_4_Callback(hObject, eventdata, handles)
% hObject    handle to Class_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Class_4


% --- Executes on button press in Class_5.
% function Class_5_Callback(hObject, eventdata, handles)
% hObject    handle to Class_5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Class_5


% --- Executes on button press in Class_6.
% function Class_6_Callback(hObject, eventdata, handles)
% hObject    handle to Class_6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Class_6


% --- Executes on button press in Class_7.
% function Class_7_Callback(hObject, eventdata, handles)
% hObject    handle to Class_7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Class_7


% --- Executes on button press in lodaButton.
function lodaButton_Callback(hObject, eventdata, handles)
% hObject    handle to lodaButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
myFolder = uigetdir;
filePattern = fullfile(myFolder, '*.JPG');
jpegFiles = dir(filePattern);

handles.imageNo = 0;
if exist(fullfile(myFolder, 'label.csv'), 'file') == 2
     m = readtable(fullfile(myFolder, 'label.csv'));
     labelNumber = size(m,1);
     imageNo = labelNumber+1;
else
    fid=fopen(fullfile(myFolder, 'label.csv'),'wt');
    x = {'FILENAME','EMPTY','SFT', 'TIMB', 'FMWK', ...
    'HP', 'SP', 'BRWC', 'CP'};
    [rows,cols]=size(x);
    for i=1:rows
          fprintf(fid,'%s,',x{i,1:end-1});
          fprintf(fid,'%s\n',x{i,end});
    end
    fclose(fid);
    fid=fopen(fullfile(myFolder, 'data.csv'),'wt');
    fclose(fid);
    imageNo = 1;
end

if imageNo <= length(jpegFiles)
     baseFileName = jpegFiles(imageNo).name;
     fullFileName = fullfile(myFolder, baseFileName);
     fprintf(1, 'Now reading %s\n', baseFileName);
     imageArray = imread(fullFileName);
     imshow(imageArray);
     handles.imageArray = imageArray;
     set(handles.startButton,'Enable','on')
else
     ax=handles.axes1;
     cla(ax)
     text(0.5, 0.5, 'All images are labeled.', 'Parent', ax);
     set(handles.startButton,'Enable','off')
end

handles.imageNo = imageNo;
handles.jpegFiles = jpegFiles;
handles.myFolder = myFolder;
guidata(hObject, handles);





% --- Executes on button press in startButton.
function startButton_Callback(hObject, eventdata, handles)
% hObject    handle to startButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
imageNo = handles.imageNo;
jpegFiles = handles.jpegFiles;
myFolder = handles.myFolder;
class = handles.class;
baseFileName = jpegFiles(imageNo).name;

Class_0 = 0;
Class_1 = 0;
Class_2 = 0;
Class_3 = 0;
Class_4 = 0;
Class_5 = 0;
Class_6 = 0;
Class_7 = 0;

switch class;
    case 0
        Class_0 = 1;
    case 1
        Class_1 = 1;
    case 2
        Class_2 = 1;
    case 3
        Class_3 = 1;
    case 4
        Class_4 = 1;
    case 5
        Class_5 = 1;
    case 6
        Class_6 = 1;
    case 7
        Class_7 = 1;
end

x = {baseFileName, Class_0, Class_1, Class_2, Class_3, Class_4, Class_5,...
    Class_6, Class_7};

labelFile = fullfile(myFolder, 'label.csv');
fileID = fopen(labelFile,'at');
formatSpec = '%s, %d, %d, %d, %d, %d, %d, %d, %d\n';
fprintf(fileID,formatSpec,x{:});
fclose(fileID);

imageArray = handles.imageArray;
imageArray = imresize(imageArray, [225 300]);
imarray = reshape(imageArray,[1,202500]);
dlmwrite(fullfile(myFolder, 'data.csv'),imarray,'-append')



imageNo = imageNo+1;
if imageNo <= length(jpegFiles)
     baseFileName = jpegFiles(imageNo).name;
     fullFileName = fullfile(myFolder, baseFileName);
     fprintf(1, 'Now reading %s\n', baseFileName);
     imageArray = imread(fullFileName);
     imshow(imageArray);
     handles.imageArray = imageArray;
else
     ax=handles.axes1;
     cla(ax)
     text(0.5, 0.5, 'You have finished labeling', 'Parent', ax);
     set(handles.startButton,'Enable','off')
end
handles.imageNo = imageNo;
guidata(hObject, handles);


% dlmwrite(fullfile(myFolder, 'label.csv'),x,...
%     'delimiter',',','-append','roffset',1);


% --- Executes when selected object is changed in uibuttongroup1.
% function uibuttongroup1_SelectionChangedFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in uibuttongroup1 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)






% function edit2_Callback(hObject, eventdata, handles)
% % hObject    handle to edit2 (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % Hints: get(hObject,'String') returns contents of edit2 as text
% %        str2double(get(hObject,'String')) returns contents of edit2 as a double


% % --- Executes during object creation, after setting all properties.
% function edit2_CreateFcn(hObject, eventdata, handles)
% % hObject    handle to edit2 (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    empty - handles not created until after all CreateFcns called
% 
% % Hint: edit controls usually have a white background on Windows.
% %       See ISPC and COMPUTER.
% if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
%     set(hObject,'BackgroundColor','white');
% end


% --- Executes when selected object is changed in uibuttongroup2.
function uibuttongroup2_SelectionChangedFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in uibuttongroup2 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
switch (get(eventdata.NewValue,'Tag'));
    case 'class_0'
        a = 0;
    case 'class_1'
        a = 1;
    case 'class_2'
        a = 2;
    case 'class_3'
        a = 3;
    case 'class_4'
        a = 4;
    case 'class_5'
        a = 5;
    case 'class_6'
        a = 6;
    case 'class_7'
        a = 7;
end
handles.class = a;
guidata(hObject, handles);


% % --- Executes on button press in class_0.
function class_0_Callback(hObject, eventdata, handles)
% % hObject    handle to class_0 (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % Hint: get(hObject,'Value') returns toggle state of class_0
