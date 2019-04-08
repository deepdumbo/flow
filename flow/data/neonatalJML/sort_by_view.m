% The previous script sorts the dicoms back into cine images and stacks,
% but still has the images sorted by study. This script will sort the
% output of the previous script and group the images by vessel/view.

in_dir = 'C:\Users\Chris\flow\data\neonatalJML\interim_1';
out_dir = 'C:\Users\Chris\flow\data\neonatalJML\interim_2';

folders = dir(in_dir);
folders(1:2) = [];

for m = 1:length(folders)  % Loop over each study
    fprintf('Sorting through study %d.\n', m);
    curr_study = [in_dir filesep folders(m).name];
    
    series = dir(curr_study);
    series(1:2) = [];
    
    for n = 1:length(series)  % Loop over each series in the study
        fprintf('..Series %d out of %d.\n', n, length(series));
        newstr = split(series(n).name, '_');
        % The first segment is the SeriesNumber. Not needed.
        if strcmp(newstr{2}, 'fl')  % Flow image
            % The vessel is in the 4th segment
            vessel = newstr{4};
            vessel = strrep(vessel, ' ', ''); % Remove spaces
            % Make the output vessel directory
            vessel_dir = [out_dir filesep vessel];
            [~, ~] = mkdir(vessel_dir);
            % List how many images already in this directory
            files = dir(vessel_dir);
            files(1:2) = [];
            num_files = length(files);
            % Matfile to copy
            dicom_info = [curr_study filesep series(n).name filesep 'dcmsinfo.mat'];
            newfilename = [vessel_dir filesep num2str(num_files+1, '%05d') '.mat'];
            % Copy
            copyfile(dicom_info, newfilename);
        else
            newstr(1) = [];
            series_desc = join(newstr, '_');
            newdir = [out_dir filesep series_desc{1}];  % Where to save
            [~, ~] = mkdir(newdir);
            % List how many images already in this directory
            files = dir(newdir);
            files(1:2) = [];
            num_files = length(files);
            % Matfile to copy
            dicom_info = [curr_study filesep series(n).name filesep 'dcmsinfo.mat'];
            newfilename = [newdir filesep num2str(num_files+1, '%05d') '.mat'];
            % Copy
            copyfile(dicom_info, newfilename);
        end
    end
end