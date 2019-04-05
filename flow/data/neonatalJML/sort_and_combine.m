% This script sorts and combines the neonatal data from Jessie.
%
% The dicoms should be sorted by study date, but within each study the
% slices and different time frames are all unsorted. This script needs to
% identify all the different slices and the frames that belong to each
% slice. Also combines magnitude and phase images into a single complex
% image.

function sort_and_combine()
%{
Main function

args:
    datadir: Root folder of dicoms
    outdir: Where to save processed outputs
%}

addpath('/home/chris/flow/utils/matlab');

datadir = '/media/chris/Data/neonatalJML/raw';
outdir = '/media/chris/Data/neonatalJML/interim';

maintime = tic;

% Get a cell array of all subfolders
folders = strsplit(genpath(datadir), ':');

% Fields from dicom to keep. I've chosen fields that seem relevant and put
% the more important ones for sorting at the beginning.
fnames = {'StudyDate';
          'SeriesTime';
          'AcquisitionTime';
          'ContentTime';
          'SeriesNumber';
          'AcquisitionNumber';
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

% Number ID for saving
s = 1;

% Loop through each folder
for m = 1:length(folders)
    looptime = tic;
    curr_folder = folders{m};
    fprintf(['Processing folder ' curr_folder '.\n']);
    % Get dicom files only
    files = dir([curr_folder filesep '*.dcm']);
    if isempty(files)
        % Skip this folder if it does not have dcm files
        fprintf('..No dicoms in this folder. Skipping.\n');
        continue;
    end
    
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
                all_info(n).(fnames{p}) = curr_info.(fnames{p}); %#ok<AGROW>
            catch
                all_info(n).(fnames{p}) = ''; %#ok<AGROW>
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
        seriestimes{n, 1} = all_info(n).SeriesTime; %#ok<AGROW>
    end
    
    % A list of the seriestimes for each cine or stack
    list_of_cine_or_stack = unique(seriestimes);
    
    % Number of cines/stacks
    num_cines = length(list_of_cine_or_stack);
    
    % Cell to hold the retrieved cines and stacks.
    cines = {};
    % Go through each cine/stack
    for n = 1:num_cines
        % Find which dicoms belong to the same cine or stack
        inds = strcmp(list_of_cine_or_stack{n}, seriestimes);
        
        % Current set of dicom information (for the cine/stack)
        cine_info = all_info(inds);
        
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
        fprintf(['..Error in similarity matrix occured in ' curr_folder '!\n']);
    end
    
    already_matched = [];
    for p = 1:num_cines
        for n = 1:num_cines
            if sim_mat(p, n)
                % If not comparing to itself
                if ~isequal(p, n)
                    % If not already matched
                    if ~ismember(p, already_matched) && ~ismember(n, already_matched)
                        % Add cines to list of already matched
                        already_matched = [already_matched; p; n];
                        % Create complex valued image data
                        cine_1_info = cines{p, 1};
                        cine_2_info = cines{n, 1};
                        
                        % (Too soon to be getting the images. Just modify
                        % this to save the dicominfo. Will need to further
                        % process the PC image to values of radians. And
                        % need to find out which one is mag and which one
                        % is phase.
                        cine_1 = {};
                        cine_2 = {};
                        for r = 1:length(cine_1_info)
                            cine_1(r).im = dicomread(cine_1_info(r).Filename);
                            cine_2(r).im = dicomread(cine_2_info(r).Filename);
                        end
                        
                        cine_1 = cat(3, cine_1.im);
                        cine_2 = cat(3, cine_2.im);
                        
                        % Which one is mag, which phase?
                        
                        save([outdir filesep num2str(s) '.mat'], 'cine_1');
                        s = s + 1;
                        save([outdir filesep num2str(s) '.mat'], 'cine_2');
                        s = s + 1;
                    end
                end
            end
        end
    end
    
    % Whichever cines were not matched, must be non-phase contrast.
    % Save them as well.
    for n = 1:num_cines
        if ~ismember(n, already_matched)
            cine_1_info = cines{n, 1};  % Dicom info for that cine
            cine_1 = {};  % To hold image data
            for r = 1:length(cine_1_info)
                cine_1(r).im = dicomread(cine_1_info(r).Filename);
            end
            cine_1 = cat(3, cine_1.im);
            % Save
            save([outdir filesep num2str(s) '.mat'], 'cine_1');
            s = s + 1;
        end
    end
    elapsedtime = toc(looptime);
    fprintf(['..Loop time: ' num2str(elapsedtime) ' s.\n']);
end

elapsedtime = toc(maintime);
fprintf(['Total time: ' num2str(elapsedtime) ' s.\n']);
end


















