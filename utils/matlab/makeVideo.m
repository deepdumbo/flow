function makeVideo(phantom)

YourData = phantom;
Uncompressed = 'n';

[r,c,n]=size(YourData); % if your CT slices are in a volume
minWindow=min(YourData(:));
maxWindow=max(YourData(:)); % or some other scale...

minWindow=0.000;
maxWindow=1.2;

iptsetpref('ImshowBorder','tight');
for i=1:n
figure1 = figure;
imshow(YourData(:,:,i),[minWindow maxWindow])
%imagesc(squeeze(YourData(:,:,i)),[minWindow maxWindow])
% colormap('gray');
% annotation(figure1,'textbox',...
%     [0.448265625 0.87837837837838 0.124 0.124999999999999],'String',num2str(i),...
%     'LineStyle','none',...
%     'FontWeight','bold',...
%     'FontSize',24,...
%     'FitBoxToText','off');

%colorbar
%title([num2str(i+9) ' keV']);
% axis tight; axis equal; axis off;
%xlabel('v'); ylabel('u'); title(colorbar, '# photons');
drawnow
F(i)=getframe(gcf); % This makes the frame of your movie
end

% Write the movie
MovieName='AwesomeCTFlyThrough.avi';
if Uncompressed == 'n'
    vidObj=VideoWriter(MovieName);
else
    vidObj=VideoWriter(MovieName,'Uncompressed AVI');
end
vidObj.FrameRate=8; % Might be too fast or too slow
open(vidObj);
writeVideo(vidObj,F);
close(vidObj);

%====================
%OR:

% % If you are reading in dicoms a slice at a time, the code might change like:
% file_list=dir('C:/WheresYourData/*.dcm');
% for i=1:n
% YourData=dicomread(file_list(i).name)
% imshow(YourData,[minWindow maxWindow])
% drawnow
% F(i)=getframe(gcf); % This makes the frame of your movie
% end
end