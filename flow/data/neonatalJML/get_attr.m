function out = get_attr(fieldname, cine_info)
% cine_info: Structure with a list of dicoms and their dicominfo

out = {};
for n = 1:length(cine_info)
    out{n, 1} = cine_info(n).(fieldname); %#ok<AGROW>
end

for n = 1:length(out)
    if ~isequal(out{1, 1}, out{n, 1})
        fprintf('There was a value that should be the same throughout the cine/stack but it was not.\n');
        fprintf(['That value was ' fieldname ', in file ']);
        cine_info(1).Filename
    end
end

out = out{n, 1};

end