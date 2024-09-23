function sub = load_TMS_v1(fname)

% fname = 'allHorizonData_v1.csv';

%% read data from spreadsheet
fid = fopen(fname);

hdr = textscan(fid,...
    '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s', ...
    1, 'delimiter', ',');
data = textscan(fid,...
    '%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f', ...
    'delimiter', ',');
fclose(fid);


%% separate data into separate subjects
R = [data{13:21}]; % changed from 13:22
C = [data{22:30}]; % changed from 23:32
RT = [data{35:42}]; %changed from 33:42

subject_list = unique(data{3});

% for sn = 1:max(data{3})
for sn = 1:size(subject_list, 1)    
    subjectID = subject_list(sn);
    
    ind = find(data{3} == subjectID);
    
    sub(sn).expt_name   = {data{1}{ind}}';
    sub(sn).replication = data{2}(ind(1));
    sub(sn).subjectID   = num2str(data{3}(ind(1)));
    sub(sn).order       = data{4}(ind(1));
    sub(sn).age         = data{5}(ind(1));
    sub(sn).iswoman     = data{6}(ind(1));
    
    sub(sn).game        = data{8}(ind);
    sub(sn).gameLength  = data{9}(ind);
    sub(sn).uc          = data{10}(ind);
    sub(sn).m1          = data{11}(ind);
    sub(sn).m2          = data{12}(ind);
    
    sub(sn).r           = R(ind,:);
    sub(sn).a           = C(ind,:);
    sub(sn).RT          = RT(ind,:);
    
end
clear RT C R

%% augment data structure
for sn = 1:length(sub)
    
    % z-scored RT
    sub(sn).RTz = (sub(sn).RT - nanmean(sub(sn).RT(:))) / nanstd(sub(sn).RT(:));
    
    % running total of how many times each bandit is played
    sub(sn).n1 = cumsum(sub(sn).a == 1,2);
    sub(sn).n2 = cumsum(sub(sn).a == 2,2);
    
    
    
    % running total of reward from each bandit
    sub(sn).R1 = cumsum(sub(sn).r.*(sub(sn).a==1),2);
    sub(sn).R2 = cumsum(sub(sn).r.*(sub(sn).a==2),2);
    
    % running observed mean for each bandit
    sub(sn).o1 = sub(sn).R1 ./ sub(sn).n1;
    sub(sn).o2 = sub(sn).R2 ./ sub(sn).n2;
    
    % is choice objectively correct?
    sub(sn).co = repmat((sub(sn).m1 > sub(sn).m2), [1 9]) .* (sub(sn).a==1) ...
        + repmat((sub(sn).m1 < sub(sn).m2), [1 9]) .* (sub(sn).a==2); %changed from [1 10]
    sub(sn).co(isnan(sub(sn).a)) = nan;
    
    % is choice a low observed mean choice? (RANDOM EXPLORATION)
    sub(sn).lm = ...
        (sub(sn).o1(:,1:end-1) < sub(sn).o2(:,1:end-1)) .* (sub(sn).a(:,2:end)==1) + ...
        (sub(sn).o1(:,1:end-1) > sub(sn).o2(:,1:end-1)) .* (sub(sn).a(:,2:end)==2);
    sub(sn).lm(sub(sn).o1(:,1:end-1)==sub(sn).o2(:,1:end-1)) = nan;
    sub(sn).lm(isnan(sub(sn).a(:,2:end))) = nan;
    % shift over so that trials line up for later
    sub(sn).lm(:,2:end+1)=sub(sn).lm(:,1:end);
    sub(sn).lm(:,1) = nan;
    
    % is choice high info choice? (DIRECTED EXPLORATION)
    sub(sn).hi = (sub(sn).n1(:,1:end-1) < sub(sn).n2(:,1:end-1)) .* (sub(sn).a(:,2:end)==1) ...
        + (sub(sn).n1(:,1:end-1) > sub(sn).n2(:,1:end-1)) .* (sub(sn).a(:,2:end)==2);
    sub(sn).hi(sub(sn).n1(:,1:end-1)==sub(sn).n2(:,1:end-1)) = nan;
    sub(sn).hi(isnan(sub(sn).a(:,2:end))) = nan;
    % shift over so that trials line up for later
    sub(sn).hi(:,2:end+1)=sub(sn).hi(:,1:end);
    sub(sn).hi(:,1) = nan;
    
    % is choice same as the last thing they did? (ALTERNATION)
    sub(sn).rep = [nan(size(sub(sn).a, 1),1) (sub(sn).a(:,1:end-1) == sub(sn).a(:,2:end))];
    
    
end
