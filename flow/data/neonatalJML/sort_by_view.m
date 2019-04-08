% The previous script sorts the dicoms back into cine images and stacks,
% but still has the images sorted by study. This script will sort the
% output of the previous script and group the images by vessel/view.
% This script also fixes stacks of cines by rearranging them in the correct
% order.

addpath('/home/chris/flow/flow/utils/matlab');
% addpath('C:\Users\Chris\flow\flow\utils\matlab');

in_dir = '/media/chris/Data/neonatalJML/interim_1';
out_dir = '/media/chris/Data/neonatalJML/interim_2';
% in_dir = 'C:\Users\Chris\flow\data\neonatalJML\interim_1';
% out_dir = 'C:\Users\Chris\flow\data\neonatalJML\interim_2';

folders = dir(in_dir);
folders(1:2) = [];

for m = 1:length(folders)  % Loop over each study
    fprintf('Sorting through study %d.\n', m);
    curr_study = [in_dir filesep folders(m).name];
    
    series = dir(curr_study);
    series(1:2) = [];
    
    for n = 1:length(series)  % Loop over each series in the study
        fprintf('..Series %d out of %d.\n', n, length(series));
        newstr = split(series(n).name, '_');  % The first segment is the SeriesNumber. Not needed.
        % Matfile to copy
        dicom_info = [curr_study filesep series(n).name filesep 'dcmsinfo.mat'];
        % Open the file to check the CardiacNumberOfImages
        cine_info = load(dicom_info); fn = fieldnames(cine_info); cine_info = cine_info.(fn{1});
        if iscell(cine_info)  % Complex image (should be flow)
            if strcmp(newstr{2}, 'fl')
                % The vessel is in the 4th segment
                vessel = newstr{4};
                vessel = strrep(vessel, ' ', ''); % Remove spaces
                % Make the output vessel directory
                vessel_dir = [out_dir filesep vessel];
            else  % Should be flow image but 'fl' not in SeriesDescription
                newstr(1) = [];
                series_desc = join(newstr, '_');
                series_desc = strrep(series_desc, ' ', ''); % Remove spaces
                % Make the output vessel directory
                vessel_dir = [out_dir filesep series_desc{1}];
            end
            [~, ~] = mkdir(vessel_dir);
            % List how many images already in this directory
            files = dir(vessel_dir);
            files(1:2) = [];
            num_files = length(files);
            
            newfilename = [vessel_dir filesep num2str(num_files+1, '%05d') '.mat'];
            % Just check the magnitude images
            cnoi = get_attr('CardiacNumberOfImages', cine_info{1});  % is a cell
            if ~isequal(cnoi, length(cine_info{1}))
                fprintf('Error in the CardiacNumberOfImages.\n');
            end
            % Copy
            copyfile(dicom_info, newfilename);
        else
            newstr(1) = [];
            series_desc = join(newstr, '_');
            series_desc = strrep(series_desc, ' ', ''); % Remove spaces
            newdir = [out_dir filesep series_desc{1}];  % Where to save
            [~, ~] = mkdir(newdir);
            % List how many images already in this directory
            files = dir(newdir);
            files(1:2) = [];
            num_files = length(files);
            
            newfilename = [newdir filesep num2str(num_files+1, '%05d') '.mat'];
            cnoi = get_attr('CardiacNumberOfImages', cine_info);
            if isequal(cnoi, 1)  % Then this is a localizer
                % Copy
                copyfile(dicom_info, newfilename);
            else  % This is a stack of cines
                % Check
                ins = [cine_info.InstanceNumber];
                if ~isequal(max(ins), cnoi)
                    fprintf('Error in stack of cines.\n');
                end
                num_slices = length(cine_info)/cnoi;
                % The cine stacks are out of order. First arrange by
                % SliceLocation.
                sln = [cine_info.SliceLocation];
                % Get indices that would sort by SliceLocation
                [~, sort_inds] = sort(sln);
                cine_info = cine_info(sort_inds);
                % Then loop through each slice
                stack = {};
                % Put each slice into cell array
                for p = 1:num_slices
                    ind1 = (p-1)*cnoi + 1;
                    ind2 = p*cnoi;
                    stack{p, 1} = cine_info(ind1:ind2);
                end
                % Loop through each slice, check, and sort
                for p = 1:num_slices
                    slice_info = stack{p, 1};
                    % Check that SliceLocation values are all the same
                    a = get_attr('SliceLocation', slice_info);
                    % Get the trigger times so that the frames of the cine are in order
                    tts = [slice_info.TriggerTime];
                    % Get indices that would sort by TriggerTime
                    [~, sort_inds] = sort(tts);
                    slice_info = slice_info(sort_inds);
                    % Check the sorting against InstanceNumber
                    for r = 1:length(slice_info)
                        if ~isequal(slice_info(r).InstanceNumber, r)
                            fprintf('Error in sorting cine.\n');
                        end
                    end
                    stack{p, 1} = slice_info;
                end
                
                % Save
                save(newfilename, 'stack');
            end
        end
    end
end