directory = './raw/AdaptationData/Chewie_CO_VR_20160929/';
files = dir([directory '*.mat']);

display(files);

for i = 1:length(files)
    filename = files(i).name;
    filepath = [directory filename];
    
    display(filepath);

    load(filepath)
    for k = 1:length(data.units)
         data.units(k).spikes = struct(data.units(k).spikes);
    end

    trials = struct(data.trials)
    data.trials = cell2struct(trials.data, trials.varDim.labels, 2)
    save(filepath, 'data');
end