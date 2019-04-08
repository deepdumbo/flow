% Look at the images sorted from sort_and_combine.m

addpath('/home/chris/flow/flow/utils/matlab');

datadir = 'C:\Users\Chris\flow\data\neonatalJML\interim_1\study_1';
seriesnumber = 1;

folders = dir(datadir);
folders(1:2) = [];

imgdir = [datadir filesep folders(seriesnumber).name];

cine_info = [imgdir filesep 'dcmsinfo.mat'];

[mag_img, phase_img, mag_info, phase_info, cine_stack, stack_info] = info2img(cine_info);

% Look at the images sorted from sort_by_view.m
datadir = '/media/chris/Data/neonatalJML/interim_2/tf2d15_SA_IPAT';
imgs = dir(datadir);
imgs(1:2) = [];

num = 1;

cine_info = [datadir filesep imgs(num).name];

[mag_img, phase_img, mag_info, phase_info, cine_stack, stack_info] = info2img(cine_info);
num = num + 1;
