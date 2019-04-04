function varargout = display_stacks(varargin)
% DISPLAY_STACKS MATLAB code for display_stacks.fig
%      DISPLAY_STACKS, by itself, creates a new DISPLAY_STACKS or raises the existing
%      singleton*.
%
%      H = DISPLAY_STACKS returns the handle to a new DISPLAY_STACKS or the handle to
%      the existing singleton*.
%
%      DISPLAY_STACKS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DISPLAY_STACKS.M with the given input arguments.
%
%      DISPLAY_STACKS('Property','Value',...) creates a new DISPLAY_STACKS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before display_stacks_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to display_stacks_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help display_stacks

% Last Modified by GUIDE v2.5 13-Dec-2012 13:50:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @display_stacks_OpeningFcn, ...
    'gui_OutputFcn',  @display_stacks_OutputFcn, ...
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


% --- Executes just before display_stacks is made visible.
function display_stacks_OpeningFcn(hObject, dummy, handles, varargin) %#ok<INUSL>
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to display_stacks (see VARARGIN)

% Choose default command line output for display_stacks
handles.output = hObject;
handles.stop_now = 0;
handles.Images=varargin(1);
if length(varargin)==1
    handles.NX={1};
else
    handles.NX=varargin(2);
end
if handles.NX{1,1}==0;
    handles.NX{1,1}=0.1;
end
% handles.Images{1,1}=squeeze(abs(handles.Images{1,1}));
handles.Images{1,1}=(abs(handles.Images{1,1}));

handles.Slice=1;
handles.a=1;
handles.b=1;%/21;
handles.recolor=0;
% Update handles structure

handles.SEGFLAG=0;
if length(varargin)>2
handles.SEGFLAG=1;
    Segment=varargin{3};
for i=1:size(handles.Images{1},3)
handles.Segment{1,i}=Segment{i};
end
end

guidata(hObject, handles);

% UIWAIT makes display_stacks wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = display_stacks_OutputFcn(hObject, eventdata, handles)  %#ok<*STOUT>
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
Images=handles.Images{1};
% if handles.a*min(Images(:))>handles.b*max(Images(:))
%     handles.a=0.95*handles.b*max(Images(:))./min(Images(:));
% end
tempI=Images(:,:,:,end);
% h=imshow(Images(:,:,1,handles.Slice),[handles.a*min(tempI(:)),handles.b*max(tempI(:))]);
h=imshow(Images(:,:,1,handles.Slice),[min(tempI(:)),max(tempI(:))]);
if handles.SEGFLAG
hold on;p=plot(handles.Segment{1}(:,1),handles.Segment{1}(:,2),'r');hold off
end
axis image
%  set(h,'Units','Pixels','Position',get(0,'ScreenSize'))

movegui(h,'center') 

% axis 'image'
% axis 'fill'
set(h,'hittest','off')

set(hObject,'name',num2str(handles.Slice));
while handles.stop_now~=1
    for loop=1:size(Images,3)
        set(hObject,'name',['Slice ',num2str(handles.Slice),' of ',num2str(size(handles.Images{1},4))]);
if sum(sum(Images(:,:,loop,handles.Slice)))~=0    
        set(h,'CData',Images(:,:,loop,handles.Slice))
if handles.SEGFLAG
        set(p,'XData',handles.Segment{loop}(:,1),'YData',handles.Segment{loop}(:,2))
end
        if handles.NX{1,1}==1;
            pause(1/(size(Images,3)/handles.NX{1,1}))
        else
            pause(handles.NX{1,1}/size(Images,3));
        end
        title([num2str(loop-1),'/',num2str(size(Images,3))])%*handles.NX{1,1}/size(Images,3))
        %         pause(5/(size(Images,3)/handles.NX{1,1}))
        
        handles=guidata(hObject);
        if handles.stop_now==1
            break
        end
        %         display(num2str(loop))
        if handles.recolor==1
%             h=imshow(Images(:,:,1,handles.Slice),[handles.a*min(Images(:)),handles.b*max(Images(:))]);
% Images=Window_CINE(Images,handles.a,handles.b);
h=imshow(Images(:,:,1,handles.Slice),[min(Images(:)),max(Images(:))]);            
% axis image
handles.recolor=0;
        end
end
    end
end
close all
% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, dummy, handles) %#ok<INUSL,DEFNU>
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.stop_now = 1;
guidata(hObject, handles);
% x=handles


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.Slice=handles.Slice-1;
if handles.Slice<1
    handles.Slice=size(handles.Images{1},4);
end
guidata(hObject, handles);

% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.Slice=handles.Slice+1;
if handles.Slice>size(handles.Images{1},4)
    handles.Slice=1;
end
guidata(hObject, handles);


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.slider1,'min',0,'max',40,'SliderStep',[0.05, 0.05]);
handles.b=(round(get(handles.slider1,'Value'))+1)/41;
handles.recolor=1;
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.slider3,'min',0,'max',40,'SliderStep',[0.05, 0.05]);
handles.a=100+50*(round(get(handles.slider3,'Value'))+2)/41;
handles.recolor=1;
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
