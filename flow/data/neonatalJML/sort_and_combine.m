% This script sorts and combines the neonatal data from Jessie's study.
%
% The dicoms should be sorted by study date, but within each study the
% slices and different time frames are all unsorted. This script needs to
% identify all the different slices and the frames that belong to each
% slice. Also combines magnitude and phase images into a single complex
% image.

% datadir: Root folder of dicoms
% outdir: Where to save processed outputs

addpath('/home/chris/flow/flow/utils/matlab');
% addpath('C:\Users\Chris\flow\flow\utils\matlab');

datadir = '/media/chris/Data/neonatalJML/raw';
outdir = '/media/chris/Data/neonatalJML/interim_1';
% datadir = 'C:\Users\Chris\flow\data\neonatalJML\raw';
% outdir = 'C:\Users\Chris\flow\data\neonatalJML\interim_1';

maintime = tic;

% Get a cell array of all subfolders
folders = strsplit(genpath(datadir), ':')';
% folders = strsplit(genpath(datadir), ';')';

% Fields from dicom to keep. I've chosen fields that seem relevant and put
% the more important ones for sorting at the beginning.
fnames = {'StudyDate';
          'SeriesTime';
          'AcquisitionTime';
          'ContentTime';
          'SeriesNumber';
          'InstanceNumber';
          'SeriesDescription';
          'TriggerTime';
          'NominalInterval';
          'CardiacNumberOfImages';
          'Rows';
          'Columns';
          'PixelSpacing';
          'ImagePositionPatient';
          'ImageOrientationPatient';
          'SliceLocation';
          'SliceThickness';
          'RepetitionTime';
          'EchoTime';
          'NumberOfAverages';
          'AcquisitionNumber';
          'ImagingFrequency';
          'EchoNumber';
          'MagneticFieldStrength';
          'SpacingBetweenSlices';
          'NumberOfPhaseEncodingSteps';
          'EchoTrainLength';
          'PercentSampling';
          'PercentPhaseFieldOfView';
          'PixelBandwidth';
          'TransmitCoilName';
          'AcquisitionMatrix';
          'InPlanePhaseEncodingDirection';
          'FlipAngle';
          'PatientPosition';
          'StudyID';
          'PatientBirthDate';
          'PatientSex';
          'AccessionNumber';
          'FileModDate';
          'FileSize';
          'Format';
          'FormatVersion';
          'Width';
          'Height';
          'BitDepth';
          'ColorType';
          'FileMetaInformationGroupLength';
          'FileMetaInformationVersion';
          'InstanceCreationDate';
          'InstanceCreationTime';
          'Modality';
          'Manufacturer';
          'ManufacturerModelName';
          'BodyPartExamined';
          'ScanningSequence';
          'SequenceVariant';
          'ScanOptions';
          'MRAcquisitionType';
          'SequenceName';
          'AngioFlag';
          'ImagedNucleus';
          'SoftwareVersion';
          'VariableFlipAngleFlag';
          'SAR';
          'dBdt';
          'SamplesPerPixel';
          'PhotometricInterpretation';
          'BitsStored';
          'HighBit';
          'PixelRepresentation';
          'SmallestImagePixelValue';
          'LargestImagePixelValue';
          'RequestingService';
          'RequestedProcedureDescription';
          'RescaleIntercept';
          'RescaleSlope';
          'RescaleType';
          'Filename'};

% Loop through each folder backwards and remove ones with no dicom files
for m = length(folders):-1:1
    curr_folder = folders{m};
    files = dir([curr_folder filesep '*.dcm']);  % Get dicom files only
    if isempty(files)
        folders(m) = [];
    end
end

% Loop through each folder
for m = 1:length(folders)
    looptime = tic;
    curr_folder = folders{m};
    fprintf('Processing folder %s. %d/%d.\n', curr_folder, m, length(folders));
    % Get dicom files only
    files = dir([curr_folder filesep '*.dcm']);
    if isempty(files)
        % Skip this folder if it does not have dcm files
        fprintf('..No dicoms in this folder. Skipping.\n');
        continue;
    end
    
    % Create study directory
    studydir = [outdir filesep 'study_' num2str(m, '%03d')];
    [~, ~] = mkdir(studydir);
    
    % Number of dicom files
    num_files = length(files);
    
    % Get the dicom info from all files and put in cell.
    %{
    all_info = {};
    for n = 1:num_files
        file = [curr_folder filesep files(n).name];
        curr_info = dicominfo(file);
        % Get fieldnames
        fnames = fieldnames(curr_info);
        for p = 1:length(fnames)
            all_info(n).(fnames{p}) = curr_info.(fnames{p}); %#ok<AGROW>
        end
    end
    %}
    
    % Get dicom info from all files. Only keeps selected fields.
    all_info = {};
    for n = 1:num_files
        file = [curr_folder filesep files(n).name];
        curr_info = dicominfo(file);
        for p = 1:length(fnames)
            try
                all_info(n).(fnames{p}) = curr_info.(fnames{p});  %#ok<*SAGROW>
            catch
                all_info(n).(fnames{p}) = ''; 
            end
        end
    end
    
    % There are localizers, time-resolved anatomy stacks, and many time-
    % resolved vessels in each folder. These images have already been
    % grouped by study (but if they are not, it may be possible to do that
    % with StudyDate). SeriesTime, SeriesNumber, and AcquisitionTime may be
    % used to group these images back together.
    
    % Use SeriesTime
    seriestimes = {};
    for n = 1:num_files
        seriestimes{n, 1} = all_info(n).SeriesTime;
    end
    
    % A list of the seriestimes for each cine or stack
    list_of_cine_or_stack = unique(seriestimes);
    
    % Number of cines/stacks
    num_cines = length(list_of_cine_or_stack);
    
    % Cell to hold the retrieved cines and stacks.
    cines = {};
    % Tags. Each cine has multiple dicom files. Tags will hold a single
    % value of the dicom tags.
    tags = struct;
    % Go through each cine/stack
    for n = 1:num_cines
        % Find which dicoms belong to the same cine or stack
        inds = strcmp(list_of_cine_or_stack{n}, seriestimes);
        
        % Current set of dicom information (for the cine/stack)
        cine_info = all_info(inds);
        
        % TriggerTime or InstanceCreationTime may be able to order the
        % frames of the cine. Also InstanceNumber would work.
        
        % Get the trigger times so that the frames of the cine are in order
        tts = [cine_info.TriggerTime];
        
        % Get indices that would sort by TriggerTime
        % (I noticed a case where one of the slices of a localizer stack
        % had a different TriggerTime than all the other slices.)
        [~, sort_inds] = sort(tts);
        
        cine_info = cine_info(sort_inds);
        
        % SliceLocation and InstanceNumber may be able to order stacks. I
        % use InstanceNumber because sortying by SliceLocation will always
        % order the stack from lowest position to highest position.
        
        % Get the SliceLocation's of this cine or stack
        % sls = [cine_info.SliceLocation];
        
        % Get the InstanceNumber's of this cine or stack
        ins = [cine_info.InstanceNumber];
        
        % Get indices that would sort by InstanceNumber
        [~, sort_inds] = sort(ins);
        
        cine_info = cine_info(sort_inds);
        
        % Store in list
        cines{n, 1} = cine_info;
        
        % Get tags
        tags(n).SeriesTime = get_attr('SeriesTime', cine_info);
        tags(n).SeriesNumber = get_attr('SeriesNumber', cine_info);
        % tags(n).InstanceNumber = get_attr('InstanceNumber', cine_info);
        tags(n).SeriesDescription = get_attr('SeriesDescription', cine_info);
        % tags(n).TriggerTime = get_attr('TriggerTime', cine_info);
        % tags(n).InstanceCreationTime = get_attr('InstanceCreationTime', cine_info);
        tags(n).PixelSpacing = get_attr('PixelSpacing', cine_info);
        % tags(n).ImagePositionPatient = get_attr('ImagePositionPatient', cine_info);
        % tags(n).ImageOrientationPatient = get_attr('ImageOrientationPatient', cine_info);
    end
    
    % The magnitude and phase images corresponding to the same slice need
    % to be re-combined into one complex image. The SliceLocation or
    % ImagePositionPatient may be able to pair these cines back together. I
    % will try ImagePositionPatient as that is more specific.
    
    % Get the ImagePositionPatient from each cine
    ipp = {};
    for n = 1:num_cines
        % Get the set of info for that cine/stack
        cine_info = cines{n, 1};
        % Get list of ImagePositionPatient for each dicom
        ipp{n, 1} = [cine_info.ImagePositionPatient]';
    end
    
    % See which cines match in terms of ImagePositionPatient.
    % The only way I can think of right now is to compare each cine with
    % every other cine, forming a 'similarity matrix'.
    sim_mat = zeros(num_cines, num_cines); % Matrix of zeros
    for p = 1:num_cines
        for n = 1:num_cines
            % If position matrices are equivalent
            if isequal(ipp{p, 1}, ipp{n, 1})
                sim_mat(p, n) = 1;
            end
        end
    end
    
    % Each column of the similarity matrix cannot sum to be greater than 2
    check = sum( sum(sim_mat) > 2 );
    if check
        fprintf('..Error in similarity matrix occured.\n');
        
        % Some series may be acquired twice at the same position, leading
        % to the error above. The TriggerTime should be different between
        % re-acquisitions. Use TriggerTime to try to pair the magnitude and
        % phase images.
        
        % Get the TriggerTime from each cine
        tts = {};
        for n = 1:num_cines
            % Get the set of info for that cine/stack
            cine_info = cines{n, 1};
            % Get list of TriggerTime for each dicom
            tts{n, 1} = [cine_info.TriggerTime]';
        end
        
        for p = 1:num_cines
            for n = 1:num_cines
                if sim_mat(p, n)  % If their positions matched
                    if isequal(tts{p, 1}, tts{n, 1})
                        % Then these should be the correct pair
                    else
                        sim_mat(p, n) = 0;
                    end
                end
            end
        end
        
        check = sum( sum(sim_mat) > 2 );
        if check
            fprintf('..Error in similarity matrix still not fixed!\n');
            % This might not actually be a problem, because in the next
            % part of the code, the cines are paired in sequential order,
            % which means it will pair adjacent SeriesNumber's. This means
            % that this problem does not have to be fixed as long as the
            % magnitude and phase cines that belong to the same acquisition
            % are adjacent. Otherwise it is very difficult to fix the
            % similarity matrix because almost all the dicom info are the
            % same between these repeat acquisitions that have same 
            % ImagePositionPatient and TirggerTime.
        else
            fprintf('..Fixed.\n');
        end
    end
    
    already_matched = [];
    for p = 1:num_cines
        for n = 1:num_cines
            if sim_mat(p, n)
                if ~isequal(p, n)  % If not comparing to itself
                    % If not already matched
                    if ~ismember(p, already_matched) && ~ismember(n, already_matched)
                        % Add cines to list of already matched
                        already_matched = [already_matched; p; n]; %#ok<*AGROW>
                        
                        % Get cine info of mag and phase
                        cine_1_info = cines{p, 1};
                        cine_2_info = cines{n, 1};
                        
                        % Use SeriesDescription to find out which one is
                        % mag and which one is the phase
                        seriesdescrip_1 = get_attr('SeriesDescription', cine_1_info);
                        seriesdescrip_2 = get_attr('SeriesDescription', cine_2_info);
                        
                        shorter = min(length(seriesdescrip_1), length(seriesdescrip_2));
                        
                        if ~strcmp(seriesdescrip_1(1:shorter), seriesdescrip_2(1:shorter))
                            % If that base part of the SeriesDescription
                            % not the same.
                            fprintf('Warning: SeriesDescription does not match\n');
                        end
                        
                        bool_1 = strcmp('_P', seriesdescrip_1(end-1:end));
                        bool_2 = strcmp('_P', seriesdescrip_2(end-1:end));
                        if bool_1 && bool_2
                            fprintf('Error. Both cines are phase images?\n');
                        elseif ~(bool_1 || bool_2)
                            fprintf('Error. Neither image was the phase image.\n');
                        elseif bool_1
                            phase_img = cine_1_info;
                            mag_img = cine_2_info;
                        elseif bool_2
                            phase_img = cine_2_info;
                            mag_img = cine_1_info;
                        else
                            % This should never happen.
                            fprintf('Catastrophic error...\n');
                        end
                        
                        % Final check... Check SeriesNumber. The
                        % SeriesNumber of the magnitude image should be one
                        % number before the phase image.
                        sn_1 = get_attr('SeriesNumber', mag_img);
                        sn_2 = get_attr('SeriesNumber', phase_img);
                        if (sn_2 - sn_1) == 1
                        else
                            fprintf('Catastrophic error type 2...\n');
                        end
                        
                        % Still just the dicominfo. Not the pixel data.
                        complex_img = {mag_img; phase_img};
                        
                        % (Too soon to be getting the images. Just modify
                        % this to save the dicominfo. Will need to further
                        % process the PC image to values of radians. And
                        % need to find out which one is mag and which one
                        % is phase.
                        %{
                        cine_1 = {};
                        cine_2 = {};
                        for r = 1:length(cine_1_info)
                            cine_1(r).im = dicomread(cine_1_info(r).Filename);
                            cine_2(r).im = dicomread(cine_2_info(r).Filename);
                        end
                        
                        cine_1 = cat(3, cine_1.im);
                        cine_2 = cat(3, cine_2.im);
                        %}
                        
                        % SequenceName, SeriesDescription,
                        % LargestImagePixelValue, RescaleIntercept,
                        % RescaleSlope, and RescaleType may be able to
                        % identify mag and phase.
                        
                        series_des = get_attr('SeriesDescription', mag_img);
                        savefolder = [studydir filesep num2str(sn_1, '%03d') '_' series_des];
                        savename = [savefolder filesep 'dcmsinfo.mat'];
                        [~, ~] = mkdir(savefolder);
                        save(savename, 'complex_img');
                    end
                end
            end
        end
    end
    
    % Whichever cines were not matched, must be non-phase contrast.
    % Save them as well.
    for n = 1:num_cines
        if ~ismember(n, already_matched)
            img = cines{n, 1};  % Dicom info for that cine
            %{
            cine_1 = {};  % To hold image data
            for r = 1:length(cine_1_info)
                cine_1(r).im = dicomread(cine_1_info(r).Filename);
            end
            cine_1 = cat(3, cine_1.im);
            %}
            % Save
            seriesnumber = get_attr('SeriesNumber', img);
            series_des = get_attr('SeriesDescription', img);
            savefolder = [studydir filesep num2str(seriesnumber, '%03d') '_' series_des];
            savename = [savefolder filesep 'dcmsinfo.mat'];
            [~, ~] = mkdir(savefolder);
            save(savename, 'img');
        end
    end
    elapsedtime = toc(looptime);
    fprintf(['..Loop time: ' num2str(elapsedtime) ' s.\n\n']);
end

elapsedtime = toc(maintime);
fprintf(['Total time: ' num2str(elapsedtime) ' s.\n']);

%{
TODO: Make a function to look at images. Sort them by vessel. Will need to check that the 
anatomy images are in order.
%}

%{
TODO: InstanceNumber may be able to sort the frames of a cine back into the
right order, making TriggerTime not needed.
%}

%{
To find the cinestacks, find images where the CardiacNumberofImages is
different than the length of the structure.
%}












