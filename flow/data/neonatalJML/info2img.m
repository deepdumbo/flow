function [mag_img, phase_img, mag_info, phase_info] = info2img(cine_info)
% cine_info: mat file containing a structure of dicom info. If it is a
% complex image, the mat file will contain a (2 x 1) cell with the first
% cell containing the structure of dicom info for the magnitude image and
% the second for the phase image.

% Standard stuff to open the file
a = load(cine_info);
fn = fieldnames(a);
a = a.(fn{1});

if iscell(a)
    % Then this is a complex image
    mag_info = a{1};
    phase_info = a{2};
    for n = 1:length(mag_info)
        % Put dicom image data into structure
        mag_img(n).im = dicomread(mag_info(n).Filename); %#ok<AGROW>
        phase_img(n).im = dicomread(phase_info(n).Filename); %#ok<AGROW>
    end
    mag_img = cat(3, mag_img.im);
    phase_img = cat(3, phase_img.im);
    %{
    figure('color', 'w'); imshow3D(mag_img);
    figure('color', 'w'); imshow3D(phase_img);
    %}
    display_stacks(mag_img);
    display_stacks(phase_img);
else
    % It is just magnitude image
    mag_info = a;
    for n = 1:length(mag_info)
        mag_img(n).im = dicomread(mag_info(n).Filename); %#ok<AGROW>
    end
    mag_img = cat(3, mag_img.im);
    %{
    figure('color', 'w'); imshow3D(mag_img);
    %}
    display_stacks(mag_img);
    phase_info = [];
    phase_img = [];
end