function [mag_img, phase_img, mag_info, phase_info, cine_stack, stack_info] = info2img(cine_info)
% cine_info: mat file containing a structure of dicom info. If it is a
% complex image, the mat file will contain a (2 x 1) cell with the first
% cell containing the structure of dicom info for the magnitude image and
% the second for the phase image.

% Standard stuff to open the file
a = load(cine_info);
fn = fieldnames(a);
a = a.(fn{1});

if iscell(a)
    if strcmp(fn{1}, 'complex_img')
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
        
        figure('color', 'w'); imshow3D(mag_img);
        figure('color', 'w'); imshow3D(phase_img);
        % display_stacks(mag_img);
        % display_stacks(phase_img);
        cine_stack = [];
        stack_info = [];
    elseif strcmp(fn{1}, 'stack')  % Otherwise should be stack of cines
        num_slices = length(a);
        stack = {};
        for n = 1:num_slices
            slice_info = a{n, 1};
            num_frames = length(slice_info);
            for m = 1:num_frames
                slice(m).im = dicomread(slice_info(m).Filename); %#ok<AGROW>
            end
            stack{n, 1} = cat(3, slice.im);  % Put cine array in stack (cell)
        end
        [nx, ny, nt] = size(stack{1, 1});
        cine_stack = zeros(nx, ny, nt, num_slices, 'uint16');
        for n = 1:num_slices
            cine_stack(:, :, :, n) = stack{n, 1};
        end
        % display_stacks(cine_stack);
        stack_info = a;
        mag_img = [];
        phase_img = [];
        mag_info = [];
        phase_info = [];
    else
        fprintf('Error. Unrecognized input!\n');
    end
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
    cine_stack = [];
    stack_info = [];
end