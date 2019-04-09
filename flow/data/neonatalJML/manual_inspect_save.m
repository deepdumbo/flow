% This script takes the sorted images in the folder interim_2, opens the
% image data, shows them, requests manual approval from user, and then
% saves them to the output folder. This is very manual, hard-coded script.

% For now just use the DAO images
in_dir = '/media/chris/Data/neonatalJML/interim_2/DAO';
out_dir = '/media/chris/Data/neonatalJML/interim_3/DAO';

files = dir(in_dir);
files(1:2) = [];

files_excluded = [];
keep_key = zeros(length(files), 1);

for n = 1:length(files)
    % Open dicom information
    file = [in_dir filesep files(n).name];
    fprintf('Reading file %s.\n', file);
    cine_info = load(file);
    fn = fieldnames(cine_info);
    cine_info = cine_info.(fn{1});
    mag_info = cine_info{1};
    phase_info = cine_info{2};
    
    % Read the image data
    mag_img = struct;
    phase_img = struct;
    for m = 1:length(mag_info)
        mag_img(m).im = dicomread(mag_info(m).Filename);
        phase_img(m).im = dicomread(phase_info(m).Filename);
    end
    mag_img = cat(3, mag_img.im);
    phase_img = cat(3, phase_img.im);
    
    % Convert the phase_img back to radians
    mag_img = single(mag_img);
    phase_img = single(phase_img);
    phase_img = (phase_img - 2048) / 2048 * pi;
    
    % Combine into complex image
    img = mag_img.*exp(1j.*phase_img);
    
    fprintf('..The image size is (%d, %d, %d).\n', size(img));
    fprintf('..The image resolution is (%f, %f).\n', (mag_info(1).PixelSpacing));
    
    showme(abs(img));
    showme(angle(img));
    
    x = input('....Enter 1 to keep or 0 to exclude.\n');
    keep_key(n) = x;
    if isequal(x, 1)
        % Save the image and info
        savename = [out_dir filesep files(n).name];
        save(savename, 'img', 'mag_info', 'phase_info');
    end
    
    
end

% Save the list of files that were excluded
sn = [out_dir filesep 'keep_key.mat'];
save(sn, 'keep_key');


