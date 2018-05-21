% This code is an example code for ref. [1].
% The code is based on the framework released by ref. [2].
% If you use this code, please cite the following two papers:
% Reference:
% [1] J. Jiang, X. Ma, C. Chen, L. Tao, Z.Wang, and J. Ma, ¡°Single Image Super-Resolution via Locally RegularizedAnchored Neighborhood Regression and Nonlocal Means,¡± IEEE Transactions on Multimedia,vol. 19, no. 1, pp. 15-26, 2017.
% [2] Radu Timofte, Vincent De Smet, Luc Van Gool. Anchored Neighborhood Regression for Fast Example-Based Super-Resolution. International Conference on Computer Vision (ICCV), 2013. 
% For any questions, email me by junjun0595@163.com

clc
clear;  
    
p = pwd;
addpath(fullfile(p, '/methods'));  % the upscaling methods
addpath(fullfile(p, '/include'));  % the upscaling methods

addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm

addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm

imgscale = 1; % the scale reference we work with
flag = 1;       % flag = 0 - only GR, ANR and bicubic methods, the other get the bicubic result by default
                % flag = 1 - all the methods are applied

ressss = cell(1,3);
for upscaling = [3]; % the magnification factor x2, x3, x4...

input_dir = 'Set5'; % Directory with input images from Set5 image dataset
% input_dir = 'Set14'; % Directory with input images from Set14 image dataset

pattern = '*.bmp'; % Pattern to process

dict_sizes = [16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536];
neighbors = [1:1:12, 16:4:32, 40:8:64, 80:16:128, 256, 512, 1024];
%d = 7
%for nn=1:28
%nn= 28

disp('The experiment corresponds to the results from Table 2 in the reference paper.');

disp(['The experiment uses ' input_dir ' dataset and aims at a magnification of factor x' num2str(upscaling) '.']);
if flag==1
    disp('All methods are employed : Bicubic, Yang et al., Zeyde et al., GR, ANR, NE+LS, NE+NNLS, and NE+LLE.');    
else
    disp('We run only for Bicubic, GR and ANR methods, the other get the Bicubic result by default.');
end

fprintf('\n\n');
Tempp = [];

for para3= [1]    
for para1 = [0.00001]
    for para2 = [12]
        
for d=7    
    tag = [input_dir '_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms'];
    
    disp(['Upscaling x' num2str(upscaling) ' ' input_dir ' with Zeyde dictionary of size = ' num2str(dict_sizes(d))]);
    
    mat_file = ['conf_Zeyde_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];    
    
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf');
    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using Zeyde approach...']);
        % Simulation settings
        conf.scale = upscaling; % scale-up factor
        conf.level = 1; % # of scale-ups to perform
        conf.window = [3 3]; % low-res. window size
        conf.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf.filters = {G, G.', L, L.'}; % 2D versions
        conf.interpolate_kernel = 'bicubic';

        conf.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf.overlap = [2 2]; % partial overlap (for faster training)
        end
        
        startt = tic;
        conf = learn_dict(conf, load_images(...            
            glob('CVPR08-SR/Data/Training', '*.bmp') ...
            ), dict_sizes(d));       
        conf.overlap = conf.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf.trainingtime = toc(startt);
        toc(startt)
        
        save(mat_file, 'conf');                       
        
        % train call        
    end
            
    if dict_sizes(d) < 1024
        lambda = 0.01;
    elseif dict_sizes(d) < 2048
        lambda = 0.1;
    elseif dict_sizes(d) < 8192
        lambda = 1;
    else
        lambda = 5;
    end
       
    if dict_sizes(d) < 10000
        conf.ProjM = inv(conf.dict_lores'*conf.dict_lores+lambda*eye(size(conf.dict_lores,2)))*conf.dict_lores';    
        conf.PP = (1+lambda)*conf.dict_hires*conf.ProjM;
    else
        % here should be an approximation
        conf.PP = zeros(size(conf.dict_hires,1), size(conf.V_pca,2));
        conf.ProjM = [];
    end
    
    conf.filenames = glob(input_dir, pattern); % Cell array  
    %conf.filenames = {conf.filenames{4}};
    
    conf.desc = {'Original', 'Bicubic', 'Yang et al.', ...
        'Zeyde et al.', 'Our GR', 'Our ANR', ...
        'LANR','LANR+NLM'};
    conf.results = {};
    
    %conf.points = [1:10:size(conf.dict_lores,2)];
    conf.points = [1:1:size(conf.dict_lores,2)];
    
    conf.pointslo = conf.dict_lores(:,conf.points);
    conf.pointsloPCA = conf.pointslo'*conf.V_pca';
    
    % precompute for ANR the anchored neighborhoods and the projection matrices for
    % the dictionary 
    
    conf.PPs = [];    
    if  size(conf.dict_lores,2) < 40
        clustersz = size(conf.dict_lores,2);
    else
        clustersz = 40;
    end
    D = abs(conf.pointslo'*conf.dict_lores);
    %D = conf.pointslo'*conf.dict_lores;
    
    for i = 1:length(conf.points)
        [vals idx] = sort(D(i,:), 'descend');
        if (clustersz >= size(conf.dict_lores,2)/2)
            conf.PPs{i} = conf.PP;
        else
            Lo = conf.dict_lores(:, idx(1:clustersz));  
            Hi = conf.dict_hires(:, idx(1:clustersz));
            conf.PPs{i} = 1.01*Hi*inv(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';              
%             conf.PPs{i} = 1*Hi*inv(Lo'*Lo + 1e-5*diag(1./(vals(1:clustersz)).^11)   )*Lo';   
        end
    end
    
    conf.PPsTMM = [];    
    if  size(conf.dict_lores,2) < 40
        clustersz = size(conf.dict_lores,2);
    else
        clustersz = 200;
    end
    D = abs(conf.pointslo'*conf.dict_lores);
    %D = conf.pointslo'*conf.dict_lores;
    
    for i = 1:length(conf.points)
        [vals idx] = sort(D(i,:), 'descend');
        if (clustersz >= size(conf.dict_lores,2)/2)
            conf.PPs{i} = conf.PP;
        else
            Lo = conf.dict_lores(:, idx(1:clustersz));  
            Hi = conf.dict_hires(:, idx(1:clustersz));           
            conf.PPsTMM{i} = 1*Hi*inv(Lo'*Lo + 1e-5*diag(1./(vals(1:clustersz)).^11)   )*Lo';   
            conf.dict_middle(:,i) = conf.PPsTMM{i}*conf.dict_lores(:,i);
        end
    end
    
    conf.result_dirImages = qmkdir([input_dir '/results_' tag]);
    conf.result_dirImagesRGB = qmkdir([input_dir '/results_' tag 'RGB']);
    conf.result_dir = qmkdir(['Results-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);
    conf.result_dirRGB = qmkdir(['ResultsRGB-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);
    
    %%
    t = cputime;    
        
    conf.countedtime = zeros(numel(conf.desc),numel(conf.filenames));
    
    res =[];
    for i = 1:numel(conf.filenames)
        f = conf.filenames{i};
        [p, n, x] = fileparts(f);
        [img, imgCB, imgCR] = load_images({f}); 
        if imgscale<1
            img = resize(img, imgscale, conf.interpolate_kernel);
            imgCB = resize(imgCB, imgscale, conf.interpolate_kernel);
            imgCR = resize(imgCR, imgscale, conf.interpolate_kernel);
        end
        sz = size(img{1});
        
        fprintf('%d/%d\t"%s" [%d x %d]\n', i, numel(conf.filenames), f, sz(1), sz(2));
    
        img = modcrop(img, conf.scale^conf.level);
        imgCB = modcrop(imgCB, conf.scale^conf.level);
        imgCR = modcrop(imgCR, conf.scale^conf.level);

            low = resize(img, conf.scale, conf.interpolate_kernel);
            if ~isempty(imgCB{1})
                lowCB = resize(imgCB, conf.scale, conf.interpolate_kernel);
                lowCR = resize(imgCR, conf.scale, conf.interpolate_kernel);
            end
            
        interpolated = resize2(low, conf.scale^conf.level, conf.interpolate_kernel);
        if ~isempty(imgCB{1})
            interpolatedCB = resize2(lowCB, conf.scale, conf.interpolate_kernel);    
            interpolatedCR = resize2(lowCR, conf.scale, conf.interpolate_kernel);    
        end
        
        res{1} = interpolated;
                        
        if (flag == 0) && (dict_sizes(d) == 1024)
            startt = tic;
            res{2} = {yima(low{1}, upscaling)};                        
            toc(startt)
            conf.countedtime(2,i) = toc(startt);
        else
            res{2} = interpolated;
        end
        
        if (flag == 0)
            startt = tic;
            res{3} = scaleup_Zeyde(conf, low);
            toc(startt)
            conf.countedtime(3,i) = toc(startt);    
        else
            res{3} = interpolated;
        end
        
        if flag == 1
            startt = tic;
            res{4} = scaleup_GR(conf, low);
            toc(startt)
            conf.countedtime(4,i) = toc(startt);    
        else
            res{4} = interpolated;
        end
        
        startt = tic;
        res{5} = scaleup_ANR(conf, low);
       
        toc(startt)
        conf.countedtime(5,i) = toc(startt);    
        
        if flag == 1
            startt = tic;
            res{6} = scaleup_LANR(conf, low);
            toc(startt)
            conf.countedtime(6,i) = toc(startt);    
        else
            res{6} = interpolated;
        end
        
        % res{7} is the results of LANR-NLM method
        if flag == 1
            startt = tic;            
            res{7} = improve_NL_DB(double(low{1,1}), double(img{1,1}), res{6}{1,1}, conf.scale, fspecial('gauss', 7, 1.6), 160, conf);
            toc(startt)
            conf.countedtime(7,i) = toc(startt);    
        else
            res{7} = interpolated;
        end
        
            
        result = cat(3, img{1}, interpolated{1}, res{2}{1}, res{3}{1}, ...
            res{4}{1}, res{5}{1}, res{6}{1}, res{7}{1});
        result = shave(uint8(result * 255), conf.border * conf.scale);
        
        if ~isempty(imgCB{1})
            resultCB = interpolatedCB{1};
            resultCR = interpolatedCR{1};           
            resultCB = shave(uint8(resultCB * 255), conf.border * conf.scale);
            resultCR = shave(uint8(resultCR * 255), conf.border * conf.scale);
        end

        conf.results{i} = {};
        for j = 1:numel(conf.desc)            
            conf.results{i}{j} = fullfile(conf.result_dirImages, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);            
            imwrite(result(:, :, j), conf.results{i}{j});

            conf.resultsRGB{i}{j} = fullfile(conf.result_dirImagesRGB, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);
            if ~isempty(imgCB{1})
                rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
                rgbImg = ycbcr2rgb(rgbImg);
            else
                rgbImg = cat(3,result(:,:,j),result(:,:,j),result(:,:,j));
            end
            
            imwrite(rgbImg, conf.resultsRGB{i}{j});
        end        
        conf.filenames{i} = f;
    end   
    conf.duration = cputime - t;

    % Test performance
    scores = run_comparison(conf);
%     process_scores_Tex(conf, scores,length(conf.filenames));
    
%     tempp = [para3 para1 para2  mean(scores(:,11)) mean(scores(:,12))];
%     Tempp = [Tempp;tempp]
    
    run_comparisonRGB(conf); % provides color images and HTML summary
%     %%    
%     save([tag '_' mat_file '_results_imgscale_' num2str(imgscale)],'conf');
end
    end
    end
end

ressss{1,upscaling-1} = scores;
end
%