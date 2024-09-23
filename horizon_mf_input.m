% Horizon Model Free

clear all

% gather info for horizon_fit_adm_meth script to fit old model and get
% model free behavior

dbstop if error

if ispc
    root = 'L:/';
elseif ismac
    root = '/Volumes/labs/';
elseif isunix 
    root = '/media/labs/';
end

study = 'METH'; % ADM or METH

    if study == "ADM"
        datadir = [root 'rsmith/adm-common/data/raw'];
        group_list = readtable("L:/rsmith/lab-members/clavalley/sandbox/adm/cobre/COBREAnxietyAndDecis-Dashboard_DATA_2023-01-06_1136.csv");
        completers = group_list.dshbrd_hrznrun2stat==1;
        group_list = group_list(completers,:);

        IGNORE = {'sub-AJ826'}; % Horizon data lost in the ether
        
    else 
        datadir = [root 'rsmith/adm-meth-pilot-common/data/raw'];
        group_list = readtable("L:/rsmith/lab-members/clavalley/sandbox/adm/meth/ADMMethPilot-DashboardReport_DATA_2023-01-06_1207.csv");
        completers = group_list.dshbrd_hrznrun2stat==1;
        group_list = group_list(completers,:);

    end 

    subjects = dir([datadir '/sub-*']);
 
    tablesubs = struct2table(subjects); 
    tablesubs = extractAfter(tablesubs.name, 4);
    tablesubs = cell2table(tablesubs); 
    group_subs = ismember(tablesubs.tablesubs, group_list.record_id);

    subjects = subjects(group_subs, :);
    subjects = {subjects.name};

session = 0;
run = 1;
results_dir = 'L:/rsmith/lab-members/clavalley/sandbox/adm/meth/model_free';

data = table;
for i=1:numel(subjects)
    
    subj = extractAfter(subjects{i},4);
    if subj == "AJ826"
        continue
    elseif subj == "BR795"
        continue
    else
    data(i,:) = horizon_fit_adm_meth(subj, session, run);
    end
end


 writetable(data, [results_dir '/all-meth-subs_run-1_old-fits_mf.csv'])
