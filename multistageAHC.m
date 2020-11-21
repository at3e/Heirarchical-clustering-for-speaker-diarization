clear all
addpath('\GMutils\');
rng(5);

dim = 19; % number of mfcc features
MAX_NUM_FEATURES = 500000;
allData = zeros(MAX_NUM_FEATURES, dim);
NUM_BLOCKS = 10; % divide into blocks of audio data
allData=load('CMU_20020319-1400_allData.mat');
allData=allData.allData;
NUM_SPK = 6;
data=allData.features; % feature vectors
fidx=allData.frameno; % frame index
spkLabel=allData.labels;
allfeatureCount = length(data);
THRESH = 0.25;
MAX_ITER = 100;
MAX_DUR = 500;
K = 10; priorThresh = 1/K;
SLEN = 100; % initial segment length
NUM_SEG = int32(allfeatureCount/ SLEN);

% divide audio data into segments
for s=1: NUM_SEG
    i = 1;
    for j=(s-1)*SLEN+1:s*SLEN
        if j>allfeatureCount
          j = allfeatureCount;
        end
        x(s, i, :) = data(j, :);
        t(s, i, :) = fidx(j, :);
        i = i+1;
    end
end

%% STAGE 1: INITIAL CLUSTERING AND MODELLING

featureCount = length(x);
block = round(featureCount/ NUM_BLOCKS);
timeStamps = [];
mogFeatures = {};
cnt = 1;

for b=1:NUM_BLOCKS
    M = {};
    I = {};
    i = 1;
    for j=(b-1)*block+1:b*block
        if j>featureCount
            j = featureCount;
        end
        M{i} = squeeze(x(j, :, :));
        I{i} = squeeze(t(j, :, :));
        i = i+1;
    end
    
    % compute KLD
    disp('computing KLD between every pair...');
    iter = 0;
    N = block;
    while iter<MAX_ITER
        D_skld = inf(N);
        
        for c1=1: N - 1
            s1 = squeeze(M{c1});
            if size(s1, 1)>=MAX_DUR
                continue
            end
            for c2 = c1 + 1: N
                s2 = squeeze(M{c2});
                if size(s2, 1)>=MAX_DUR
                    continue
                end
                D_skld(c1, c2) = KLD(s1, s2, dim);
            end
        end
        
        min_d = min(D_skld, [], 'all');
        max_d = max(D_skld(~isinf(D_skld)), [], 'all');
        
        D_skld = D_skld./ (max_d - min_d);
        [abs_min, min_id] = min(D_skld(: ));
        
        disp('kld: ');
        disp(abs_min);
        
        if abs_min<THRESH
            [r,c]=ind2sub(size(D_skld),min_id);
            % merge segments
            temp = [M{r}; M{c}];
            M{r} = temp;
            M(c) = [];
            I{r} = [I{r} I{c}];
            I(c) = [];
        else
            disp('KLD above threshold value ...');
            break;           
        end
        N = length(M);
        iter = iter + 1;
    end
    timeStamps = [timeStamps; I'];
    
    % model each of the merged states as a mixture of Gaussians
    
    for m = 1: length(M)
        j = 1;
        temp = [];
        v = M{m};
        model_init = moginit(v, K);
        model = mogem(v, model_init);
        
        means = model.mean;
        priors = model.prior;
        
        for k = 1:K
            if (priors(k)>=priorThresh)
                temp(j, :) =  means(k, :);
                j = j + 1;
            end
        end
        
        mogFeatures{cnt} = [temp];
        cnt = cnt + 1;
    end
    
end

%% STAGE 2: MERGING AND RE-CLUSTERING

N = length(mogFeatures);
I = timeStamps;
n = 0;
THRESH = 0.06;
THRESH_INCR = 0.005;
DUR = 2500;
DUR_INCR = 500;
M = mogFeatures;

while n<10
    MAX_DUR = DUR + n * DUR_INCR;
    iter = 1;
    while iter<MAX_ITER
        D_skld = inf(N);
        
        for c1=1: N - 1
            s1 = squeeze(M{c1});
            if size(I{c1}, 2)>=MAX_DUR
                continue
            end
            for c2 = c1 + 1: N
                s2 = squeeze(M{c2});
                if size(I{c2}, 2)>=MAX_DUR
                    continue
                end
                D_skld(c1, c2) = KLD(s1, s2, dim);
            end
        end
        
        min_d = min(D_skld, [], 'all');
        max_d = max(D_skld(~isinf(D_skld)), [], 'all');
        D_skld = D_skld./ (max_d - min_d);
        [abs_min, min_id] = min(D_skld(: ));
        disp('kld: ');
        disp(abs_min);
        
        if abs_min<(THRESH + n * THRESH_INCR)
            [r,c]=ind2sub(size(D_skld),min_id);            
            %       merge segments
            temp = [M{r}; M{c}];
            M{r} = temp;
            M(c) = [];
            I{r} = [I{r} I{c}];
            I(c) = [];
        else
            disp('KLD above threshold value ...');
            break;
            
        end
        N = length(M); disp(N);
        iter = iter + 1;
    end
    n = n + 1;
    
    % find number of significantly large clusters
    MIN_DUR = 500;
    m = 0;
    for i = 1: N
        if size(I{i}, 2)>MIN_DUR
            m = m + 1;
        end
    end
    
    if m<=NUM_SPK
        break;
    end
end

%% STAGE 3: GENERATING SPEAKER MODELS
% find number of significantly large clusters
MIN_DUR = 500;
S = [];
m = 0;

for i = 1: N
    if size(I{i}, 2)>MIN_DUR
        m = m + 1;
        S(m , 1) = i;
    end
end

Ns = length(S);
Spk_data = cell(Ns, 1);
nSpk_data = cell(N-Ns, 1);
Spk_frames = cell(Ns, 1);
nSpk_frames = cell(N-Ns, 1);
i = 1; j = 1; k = 1;
for i = 1: N
    ind = find(S==i);
    if ~isempty(ind)
        Spk_frames{j, 1} = I{i, 1};
        j = j + 1;
    else
        nSpk_frames{k, 1} = I{n, 1};
        k = k + 1;
    end
    
end


% accumulate features belonging to speaker models
for i = 1: Ns
   t =  Spk_frames{i, 1};
   lent = length(t);
   temp = zeros(lent, dim);
   for j =1: lent
       temp(j, :) = data(find(fidx==t(j)), :);
   end
   Spk_data{i, 1} = temp;
   model_init = (temp, K);
   Spk_model{i} = mogem(temp, model_init);
end

% accumulate remaining features 
for i = 1: N-Ns
   t =  nSpk_frames{i, 1};
   lent = length(t);
   temp = zeros(lent, dim);
   for j =1: lent
       temp(j, :) = data(find(fidx==t(j)), :);
   end
   nSpk_data{i, 1} = temp;
   
   p = zeros(lent, Ns);
   for k = 1: Ns
     p(:, k) = moglogp(temp, Spk_model{k});
   end
   
   [~, pth] = max(p, [], 2);   
   n = mode(pth);
   
   Spk_frames{n, 1} = [Spk_frames{n, 1}, nSpk_frames{i, 1}];
end

%% Prepare speaker Labels

SpkLabels = [];
for s = 1:Ns
   timeframes = Spk_frames{s, 1}';
   lent = size(timeframes, 1);
   L = s*ones(size(timeframes));
   SpkLabels = [SpkLabels;L];
end


%% write into rttm file

file_name = 'CMU_20020319-1400SysOut';
eval_name = 'CMU_20020319-1400';
extension = '.rttm';
rttm_file = fopen(strcat(file_name, extension), 'w');

allStateSequence = SpkLabels;
numSpk = length(unique(SpkLabels));
for s = 1:numSpk
    spkr_name = strcat(eval_name, '_', num2str(s));
    fprintf(rttm_file, 'SPKR-INFO %s 1 <NA> <NA> <NA> unknown %s <NA>\n', eval_name, spkr_name);
end

allCount = length(SpkLabels);
for s = 1:numSpk
    spkr_name = strcat(eval_name, '_', num2str(s));
    start = 0; dur = 0;
    for k = 1:allCount(1, 1)
        disp(k);
        if (allStateSequence(k, 1) ~= s && dur > 0) || k == allCount
            % write into RTTM file
            fprintf(rttm_file, 'SPEAKER %s 1 %f %f <NA> <NA> %s <NA>\n', eval_name, start * 0.010 , dur * 0.010, spkr_name);
            dur = 0;
        end
        
        if allStateSequence(k, 1) == s && dur == 0
            start = fidx(k);
        end
        
        if allStateSequence(k, 1) == s
            dur = dur + 1;
        end
    end
end
fclose(rttm_file);