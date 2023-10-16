
clc;
clear;
% 
% srcDir = '.\maritime-defog-main_original\maritime-defog-main\Test_Data\';  % e.g., 'C:\images\foggy'
% 
% destDir = './cleaned_test_data/';     
% if ~exist(destDir, 'dir')
%     mkdir(destDir);
% end
% 
% imgFiles = dir(fullfile(srcDir, '*.jpg'));
% 
% for i = 1:length(imgFiles)
% 
%     imgPath = fullfile(imgFiles(i).folder, imgFiles(i).name);
%     I = imread(imgPath);
% 
%     I_defogged = Defogging(I);  
% 
%     fprintf('Processed image %d of %d: %s\n', i, length(imgFiles), imgFiles(i).name);
% end
% 
% fprintf('All images processed and saved in %s\n', destDir);


foggy_image = imread('./V_07_01_0016.jpg');
defogged_image = Defogging(foggy_image);
imshow(defogged_image)
% imsave
function [result] = Defogging ( input )

adjust_fog_removal = 2; 
brightness = 0.5; 

input = im2double(input);
alpha = 20000;
beta = 0.1;
gamma = 10;

[~, haze_level, ~] = parameter_sel(input);
if haze_level == 0.01
    ii = 7;
end
if haze_level == 0.001
    ii = 5 ;
end
[F, G, ~] = Decomposition(input,alpha,ii,beta,gamma);
%% Dehaze the image
A = estimate_airlight(adjust_fog_removal,F);
A = reshape(A,1,1,3);
[fog_free_layer, ~] = non_local_dehazing(F,A);
%% Compensation
Gm = Compensation(fog_free_layer,G);
result = fog_free_layer +  brightness*Gm;

gray = rgb2gray(im2uint8(result));
if median(gray(:)) < 128
    result = fog_free_layer + Gm;
end

end

function [alpha, beta, pro] = parameter_sel(img_hazy)
I = img_hazy;
gray = max(I,[],3);
%gray = adjust(gray);
f = fspecial('gaussian',10,10);
f3 = [0, -1, 0; 
      -1, 4, -1;
      0, -1, 0];
shap = abs(imfilter(gray,f3));
shap = shap(10:end-10,10:end-10);
shap = minmaxfilt(shap,20,'max','same');
gray = imfilter(gray,f);
gray = imfilter(gray,f);
gray = imfilter(gray,f);
gray = imfilter(gray,f);
gray = gray(10:end-10,10:end-10);

ratio = sum(gray(:))/sum(shap(:));
alpha = 50;
beta = 0.001;
pro = 1.6;
if(ratio>2.3)
    beta = 0.01;
    pro = 2.3;
end
end
function [varargout] = minmaxfilt(A, window, outtype, shape)
if nargin<2 || isempty(window)
    window = 3;
end
% Default output types, both min and max
if nargin<3 || isempty(outtype)
    outtype = 'both'; % min/max
end
outtype = lower(strtrim(outtype));
% Check if OUTTYPE is correct
if isempty(strmatch(outtype,{'both' 'minmax' 'maxmin' 'min' 'max'}))
    error('MINMAXFILT: unknown outtype %s', outtype);
end

% Default output types, both min and max
if nargin<4 || isempty(shape)
    shape = 'valid'; % shape
end
shape = lower(strtrim(shape));
% Check if SHAPE is correct
shapeloc = strmatch(shape,{'valid' 'same' 'full'});
if isempty(shapeloc)
    error('MINMAXFILT: unknown shape %s', shape);
end

% We do not support SPARSE
if issparse(A)
    error('MINMAXFILT: first output A must be full matrix')
end

szA = size(A);
nd = ndims(A);
% extend window size
if isscalar(window) % same for all dimensions
    window(1:nd) = min(szA,window); % Bug correced Dec 1th 2009
else
    % pad missing window size
    window(end+1:nd) = 1;
end

out = cell(1,0);
idx = cell(1,0);
nout = 0;

% MINVAL
if ~strcmp(outtype,'max')
    minval = A;
    % Create linear index array
    minidx = zeros(szA,'double');
    minidx(:) = 1:numel(A);    
    sz = size(minval);
    % Loop on dimension
    for dim=1:nd
        % Reshape minval in 3D arrays, working dimension is the middle
        % That is the form required by LEMIRE_ND_ENGINE
        p = prod(sz(1:dim-1)); % return 1 if empty
        n = sz(dim);
        q = prod(sz(dim+1:end)); % return 1 if empty
        minval = reshape(minval,[p n q]);
        win = window(dim);
        % call mex engine
        if win~=1
            disp('job done min')
            % print("I am here min")
            [minval, minidx] = lemire_nd_minengine(minval, minidx, win, shapeloc);
        end
        % Blow back to n-dimensional
        sz(dim) = size(minval,2);
        minval = reshape(minval, sz);
        minidx = reshape(minidx, sz);
    end
    nout=nout+1;
    out{nout} = minval;
    idx{nout} = minidx;
end

% MAXVAL
if ~strcmp(outtype,'min')
    maxval = A;
    % Create linear index array
    maxidx = zeros(szA,'double');
    maxidx(:) = 1:numel(A);
    sz = size(maxval);
  
    % Loop on dimension
    for dim=1:nd
        % Reshape maxval in 3D arrays, working dimension is the middle
        % That is the form required by LEMIRE_ND_ENGINE
        p = prod(sz(1:dim-1)); % return 1 if empty
        n = sz(dim);
        q = prod(sz(dim+1:end)); % return 1 if empty
        maxval = reshape(maxval,[p n q]);
        win = window(dim);
        % call mex engine
        if win~=1
            disp(['size(maxval) = ' mat2str(size(maxval)) ...
      ', size(maxidx) = ' mat2str(size(maxidx)) ...
      ', win = ' mat2str(win) ...
      ', shapeloc = ' mat2str(shapeloc)]);

            % disp('job done')
            % size(maxval), size(maxidx)
            % print("I am here max")
            [maxval ,maxidx] = lemire_nd_maxengine(maxval, maxidx, win, shapeloc);

            disp(['size(maxval)_new = ' mat2str(size(maxval)) ...
      ', size(maxidx)_new = ' mat2str(size(maxidx)) ...
      ', win = ' mat2str(win) ...
      ', shapeloc = ' mat2str(shapeloc)]);
        end
        % Blow back to n-dimensional
        sz(dim) = size(maxval,2);       
        maxval = reshape(maxval, sz);
        maxidx = reshape(maxidx, sz);
    end
    nout=nout+1;
    out{nout} = maxval;
    idx{nout} = maxidx;
end

% Assign the output
if nargout <= length(out)
    varargout = out;
else
    out = [out idx];
    varargout = out;
end

end % minmaxfilt


function [maxval, maxidx] = lemire_nd_maxengine(A, idx, window, shapeflag)

    % Inputs validation
    if isequal(size(A), size(idx))
        disp('The dimensions of the two arrays are the same.');
    else
        sz = size(A);
        idx = reshape(idx, [1, sz(2), sz(3)]);
    end
    
    % reshape(two_dim_matrix, [1, 512, 512]);
    if nargin < 4
        error('Not enough input arguments.');
    end
    [p, n, q] = size(A);
    [pi, ni, qi] = size(idx);
    
    if pi ~= p || ni ~= n
        error('A and idx must have the same first two dimensions.');
    end
    if isempty(qi)
        qi = 1;
    end
    if q ~= qi
        error('A and idx must have the same third dimension or idx should not have a third dimension.');
    end

    % Initialize outputs
    maxval = zeros(size(A));
    maxidx = zeros(size(idx));

    % Looping over the first and third dimensions
    for j = 1:q
        for k = 1:p
            a = A(k, :, j);
            current_idx = idx(k, :, min(j, qi));
            
            % Initialize wedge and other variables
            nWedge = 0;
            Wedgefirst = 1;
            Wedgelast = 0;
            left = -window;
            Wedge = zeros(1, n);
            
            % Loop over the second dimension
            for i = 1:n
                left = left + 1;
                
                % Update the wedge
                while nWedge > 0 && a(Wedge(Wedgelast)) <= a(i)
                    nWedge = nWedge - 1;
                    Wedgelast = Wedgelast - 1;
                end
                if nWedge > 0 && Wedge(Wedgefirst) <= left - window
                    nWedge = nWedge - 1;
                    Wedgefirst = Wedgefirst + 1;
                end
                nWedge = nWedge + 1;
                Wedgelast = Wedgelast + 1;
                Wedge(Wedgelast) = i;
                
                % Retrieve the max value and its index
                if i >= window
                    maxval(k, i-window+1, j) = a(Wedge(Wedgefirst));
                    maxidx(k, i-window+1, j) = current_idx(Wedge(Wedgefirst));
                end
            end
        end
    end
    
    % Handle the shapeflag
    if shapeflag == 1 % valid
        maxval = maxval(:, window:end, :);
        maxidx = maxidx(:, window:end, :);
    elseif shapeflag == 3 % full
        maxval = [zeros(p, window-1, q); maxval];
        maxidx = [zeros(p, window-1, q); maxidx];
    end
end

function [ Aout ] = estimate_airlight( gamma,img, Amin, Amax, N, spacing, K, thres )

if ~exist('thres','var') || isempty(thres), thres = 0.01 ; end
if ~exist('spacing','var') || isempty(spacing), spacing = 0.02 ; end %1/M in the paper
if ~exist('n_colors','var') || isempty(N), N = 1000 ; end %number of colors clusters
if ~exist('K','var') || isempty(K), K = 40 ; end %number of angles

if ~exist('Amin','var') || isempty(Amin), Amin = [0,0.05,0.1]; end
if ~exist('Amax','var') || isempty(Amax), Amax = 1; end

if isscalar(Amin), Amin = repmat(Amin,1,3); end 
if isscalar(Amax), Amax = repmat(Amax,1,3); end

img = impyramid(img,'reduce');
img = img.^gamma;
[img_ind, points] = rgb2ind(img, N);
[h,w,~] = size(img);
% Remove empty clusters
idx_in_use = unique(img_ind(:));
idx_to_remove = setdiff(0:(size(points,1)-1),idx_in_use);
points(idx_to_remove+1,:) = [];
img_ind_sequential = zeros(h,w);
for kk = 1:length(idx_in_use)
    img_ind_sequential(img_ind==idx_in_use(kk)) = kk;
end
% Now the min value of img_ind_sequential is 1 rather then 0, and the indices
% correspond to points

% Count the occurences if each index - this is the clusters' weight
[points_weight,~] = histcounts(img_ind_sequential(:),size(points,1));
points_weight = points_weight./(h*w);
if ~ismatrix(points), points = reshape(points,[],3); end % verify dim

%% Define arrays of candidate air-light values and angles
angle_list = reshape(linspace(0, pi, K),[],1);
% Use angle_list(1:end-1) since angle_list(end)==pi, which is the same line
% in 2D as since angle_list(1)==0
directions_all = [sin(angle_list(1:end-1)) , cos(angle_list(1:end-1)) ];

% Air-light candidates in each color channel
ArangeR = Amin(1):spacing:Amax(1);
ArangeG = Amin(2):spacing:Amax(2);
ArangeB = Amin(3):spacing:Amax(3);

%% Estimate air-light in each pair of color channels
% Estimate RG
Aall = generate_Avals(ArangeR, ArangeG);
[~, AvoteRG] = vote_2D(points(:,1:2), points_weight, directions_all, Aall, thres );
% Estimate GB
Aall = generate_Avals(ArangeG, ArangeB);
[~, AvoteGB] = vote_2D(points(:,2:3), points_weight, directions_all, Aall, thres );
% Estimate RB
Aall = generate_Avals(ArangeR, ArangeB);
[~, AvoteRB] = vote_2D(points(:,[1,3]), points_weight, directions_all, Aall, thres);

%% Find most probable airlight from marginal probabilities (2D arrays)
% Normalize (otherwise the numbers are quite large)
max_val = max( [max(AvoteRB(:)) , max(AvoteRG(:)) , max(AvoteGB(:)) ]);
AvoteRG2 = AvoteRG./max_val;
AvoteGB2 = AvoteGB./max_val;
AvoteRB2 = AvoteRB./max_val;
% Generate 3D volumes from 3 different 2D arrays
A11 = repmat( reshape(AvoteRG2, length(ArangeG),length(ArangeR))', 1,1,length(ArangeB));
tmp = reshape(AvoteRB2, length(ArangeB),length(ArangeR))';
A22 = repmat(reshape(tmp, length(ArangeR),1,length(ArangeB)) , 1,length(ArangeG),1);
tmp2 = reshape(AvoteGB2, length(ArangeB),length(ArangeG))';
A33 = repmat(reshape(tmp2, 1, length(ArangeG),length(ArangeB)) , length(ArangeR),1,1);
AvoteAll = A11.*A22.*A33;
[~, idx] = max(AvoteAll(:));
[idx_r,idx_g,idx_b] = ind2sub([length(ArangeR),length(ArangeG),length(ArangeB)],idx);
Aout = [ArangeR(idx_r), ArangeG(idx_g), ArangeB(idx_b)];


end % function estimate_airlight_2D

%% Sub functions

function Aall = generate_Avals(Avals1, Avals2)
%Generate a list of air-light candidates of 2-channels, using two lists of
%values in a single channel each
%Aall's length is length(Avals1)*length(Avals2)
Avals1 = reshape(Avals1,[],1);
Avals2 = reshape(Avals2,[],1);
A1 = kron(Avals1, ones(length(Avals2),1));
A2 = kron(ones(length(Avals1),1), Avals2);
Aall = [A1, A2];
end % function generate_Avals

function [Aout, Avote2] = vote_2D(points, points_weight, directions_all, Aall, thres)
n_directions = size(directions_all,1);
accumulator_votes_idx = false(size(Aall,1), size(points,1), n_directions);
for i_point = 1:size(points,1)
    for i_direction = 1:n_directions
		 % save time and ignore irelevant points from the get-go
        idx_to_use = find( (Aall(:, 1) > points(i_point, 1)) & (Aall(:, 2) > points(i_point, 2)));
        if isempty(idx_to_use), continue; end
		
        % calculate distance between all A options and the line defined by
        % i_point and i_direction. If the distance is smaller than a thres,
        % increase the cell in accumulator
        dist1 = sqrt(sum([Aall(idx_to_use, 1)-points(i_point, 1), Aall(idx_to_use, 2)-points(i_point, 2)].^2,2));
        %dist1 = dist1 - min(dist1);
        dist1 = dist1./sqrt(2) + 1;
        
        dist =  -points(i_point, 1)*directions_all(i_direction,2) + ...
            points(i_point, 2)*directions_all(i_direction,1) + ...
            Aall(idx_to_use, 1)*directions_all(i_direction,2) - ...
            Aall(idx_to_use, 2)*directions_all(i_direction,1);
        idx = abs(dist)<2*thres.*dist1;
        if ~any(idx), continue; end

        idx_full = idx_to_use(idx);
        accumulator_votes_idx(idx_full, i_point,i_direction) = true;
    end
end
% use only haze-lined that are supported by 2 points or more
accumulator_votes_idx2 = (sum(uint8(accumulator_votes_idx),2))>=2; 
accumulator_votes_idx = bsxfun(@and, accumulator_votes_idx ,accumulator_votes_idx2);
accumulator_unique = zeros(size(Aall,1),1);
for iA = 1:size(Aall,1)
    idx_to_use = find(Aall(iA, 1) > points(:, 1) & (Aall(iA, 2) > points(:, 2)));
    points_dist = sqrt((Aall(iA,1) - points(idx_to_use,1)).^2+(Aall(iA,2) - points(idx_to_use,2)).^2);
    points_weight_dist = points_weight(idx_to_use).*(5.*exp(-reshape(points_dist,1,[]))+1); 
    accumulator_unique(iA) = sum(points_weight_dist(any(accumulator_votes_idx(iA,idx_to_use,:),3)));
end
[~, Aestimate_idx] = max(accumulator_unique);
Aout = Aall(Aestimate_idx,:);
Avote2 = accumulator_unique; 

end % function vote_2D

function [img_dehazed, transmission] = non_local_dehazing(img_hazy, air_light)
%The core implementation of "Non-Local Image Dehazing", CVPR 2016
% 
% The details of the algorithm are described in the paper: 
% Non-Local Image Dehazing. Berman, D. and Treibitz, T. and Avidan S., CVPR2016,
% which can be found at:
% www.eng.tau.ac.il/~berman/NonLocalDehazing/NonLocalDehazing_CVPR2016.pdf
% If you use this code, please cite the paper.
%
%   Input arguments:
%   ----------------
%	img_hazy     - A hazy image in the range [0,255], type: uint8
%	air_light    - As estimated by prior methods, normalized to the range [0,1]
%	gamma        - Radiometric correction. If empty, 1 is assumed
%
%   Output arguments:
%   ----------------
%   img_dehazed  - The restored radiance of the scene (uint8)
%   transmission - Transmission map of the scene, in the range [0,1]
%
% Author: Dana Berman, 2016. 
%
% The software code of the Non-Local Image Dehazing algorithm is provided
% under the attached LICENSE.md


%% Validate input
[h,w,n_colors] = size(img_hazy);
% if (n_colors ~= 3) % input verification
%     error(['Non-Local Dehazing reuires an RGB image, while input ',...
%         'has only ',num2str(n_colors),' dimensions']);
% end

% if ~exist('air_light','var') || isempty(air_light) || (numel(air_light)~=3)
%     error('Dehazing on sphere requires an RGB airlight');
% end

% if ~exist('gamma','var') || isempty(gamma), gamma = 1; end

% img_hazy = im2double(img_hazy);
%img_hazy_corrected = img_hazy.^gamma; % radiometric correction
img_hazy_corrected = img_hazy;

%% Find Haze-lines
% Translate the coordinate system to be air_light-centric (Eq. (3))
dist_from_airlight = double(zeros(h,w,n_colors));
for color_idx=1:n_colors
    dist_from_airlight(:,:,color_idx) = img_hazy_corrected(:,:,color_idx) - air_light(:,:,color_idx);
end

% Calculate radius (Eq. (5))
radius = sqrt( dist_from_airlight(:,:,1).^2 + dist_from_airlight(:,:,2).^2 +dist_from_airlight(:,:,3).^2 );

% Cluster the pixels to haze-lines
% Use a KD-tree impementation for fast clustering according to their angles
dist_unit_radius = reshape(dist_from_airlight,[h*w,n_colors]);
dist_norm = sqrt(sum(dist_unit_radius.^2,2));
dist_unit_radius = bsxfun(@rdivide, dist_unit_radius, dist_norm);
 n_points = 1000;
% load pre-calculated uniform tesselation of the unit-sphere
fid = fopen(['TR',num2str(n_points),'.txt']);
points = cell2mat(textscan(fid,'%f %f %f')) ;
fclose(fid);
mdl = KDTreeSearcher(points);
ind = knnsearch(mdl, dist_unit_radius);


%% Estimating Initial Transmission

% Estimate radius as the maximal radius in each haze-line (Eq. (11))
K = accumarray(ind,radius(:),[n_points,1],@max);
radius_new = reshape( K(ind), h, w);
    
% Estimate transmission as radii ratio (Eq. (12))
transmission_estimation = radius./radius_new;

% Limit the transmission to the range [trans_min, 1] for numerical stability
trans_min = 0.1;
transmission_estimation = min(max(transmission_estimation, trans_min),1);


%% Regularization

% Apply lower bound from the image (Eqs. (13-14))
trans_lower_bound = 1 - min(bsxfun(@rdivide,img_hazy_corrected,reshape(air_light,1,1,3)) ,[],3);
transmission_estimation = max(transmission_estimation, trans_lower_bound);
 
% Solve optimization problem (Eq. (15))
% find bin counts for reliability - small bins (#pixels<50) do not comply with 
% the model assumptions and should be disregarded
bin_count       = accumarray(ind,1,[n_points,1]);
bin_count_map   = reshape(bin_count(ind),h,w);
bin_eval_fun    = @(x) min(1, x/50);

% Calculate std - this is the data-term weight of Eq. (15)
K_std = accumarray(ind,radius(:),[n_points,1],@std);
radius_std = reshape( K_std(ind), h, w);
radius_eval_fun = @(r) min(1, 3*max(0.001, r-0.1));
radius_reliability = radius_eval_fun(radius_std./max(radius_std(:)));
data_term_weight   = bin_eval_fun(bin_count_map).*radius_reliability;
lambda = 0.1;
transmission = wls_optimization(transmission_estimation, data_term_weight, img_hazy, lambda);


%% Dehazing
% (Eq. (16))
img_dehazed = zeros(h,w,n_colors);
leave_haze = 1; % leave a bit of haze for a natural look (set to 1 to reduce all haze)

for color_idx = 1:3
    img_dehazed(:,:,color_idx) = ( img_hazy_corrected(:,:,color_idx) - ...
 (1-leave_haze.*transmission).*air_light(color_idx) )./ max(transmission,trans_min);

end

% Limit each pixel value to the range [0, 1] (avoid numerical problems)
img_dehazed(img_dehazed > 1) = 1;
img_dehazed(img_dehazed < 0) = 0;
%img_dehazed = img_dehazed.^(1/gamma); % radiometric correction

% For display, we perform a global linear contrast stretch on the output, 
% clipping 0.5% of the pixel values both in the shadows and in the highlights 
% adj_percent = [0.005, 0.9];
% img_dehazed = adjust(img_dehazed,adj_percent);

%img_dehazed = im2uint8(img_dehazed);

end % function non_local_dehazing

function out = wls_optimization(in, data_weight, guidance, lambda)
%Weighted Least Squares optimization solver.
% Given an input image IN, we seek a new image OUT, which, on the one hand,
% is as close as possible to IN, and, at the same time, is as smooth as
% possible everywhere, except across significant gradients in the hazy image.
%
%  Input arguments:
%  ----------------
%  in             - Input image (2-D, double, N-by-M matrix).   
%  data_weight    - High values indicate it is accurate, small values
%                   indicate it's not.
%  guidance       - Source image for the affinity matrix. Same dimensions
%                   as the input image IN. Default: log(IN)
%  lambda         - Balances between the data term and the smoothness
%                   term. Increasing lambda will produce smoother images.
%                   Default value is 0.05 
%
% This function is based on the implementation of the WLS Filer by Farbman,
% Fattal, Lischinski and Szeliski, "Edge-Preserving Decompositions for 
% Multi-Scale Tone and Detail Manipulation", ACM Transactions on Graphics, 2008
% The original function can be downloaded from: 
% http://www.cs.huji.ac.il/~danix/epd/wlsFilter.m


small_num = 0.00001;

if ~exist('lambda','var') || isempty(lambda), lambda = 0.05; end

[h,w,~] = size(guidance);
k = h*w;
guidance = rgb2gray(guidance);

% Compute affinities between adjacent pixels based on gradients of guidance
dy = diff(guidance, 1, 1);
dy = -lambda./(sum(abs(dy).^2,3) + small_num);
dy = padarray(dy, [1 0], 'post');
dy = dy(:);

dx = diff(guidance, 1, 2); 
dx = -lambda./(sum(abs(dx).^2,3) + small_num);
dx = padarray(dx, [0 1], 'post');
dx = dx(:);


% Construct a five-point spatially inhomogeneous Laplacian matrix
B = [dx, dy];
d = [-h,-1];
tmp = spdiags(B,d,k,k);

ea = dx;
we = padarray(dx, h, 'pre'); we = we(1:end-h);
so = dy;
no = padarray(dy, 1, 'pre'); no = no(1:end-1);

D = -(ea+we+so+no);
Asmoothness = tmp + tmp' + spdiags(D, 0, k, k);

% Normalize data weight
data_weight = data_weight - min(data_weight(:)) ;
data_weight = 1.*data_weight./(max(data_weight(:))+small_num);

% Make sure we have a boundary condition for the top line:
% It will be the minimum of the transmission in each column
% With reliability 0.8
reliability_mask = data_weight(1,:) < 0.6; % find missing boundary condition
in_row1 = min( in,[], 1);
data_weight(1,reliability_mask) = 0.8;
in(1,reliability_mask) = in_row1(reliability_mask);

Adata = spdiags(data_weight(:), 0, k, k);

A = Adata + Asmoothness;
b = Adata*in(:);

% Solve
% out = lsqnonneg(A,b);
out = A\b;
out = reshape(out, h, w);
end

function [F, G, N] = Decomposition( I, alpha ,ii , beta, gamma)
% Layer Separation using Relative Smoothness (specific for intrinsic images)

% Image properties
I(I>1) = 1;
gray = rgb2gray(I);
[H,W,D] = size(I);

% Convolutional kernels
f1 = [1, -1];
f2 = [1; -1];
f4 = [0, -1,  0; -1,  4, -1; 0, -1,  0];

% Enhance gradient of I
[weight_x, weight_y] = Gradient_weight(I);

I_filt = imgaussfilt(gray,10);
delta_I = I - I_filt;

% main
otfFx = psf2otf(f1, [H,W]);
otfFy = psf2otf(f2, [H,W]);
otfL  = psf2otf(f4, [H,W]);

fft_double_laplace = abs(otfL).^2 ;
fft_double_grad    = abs(otfFx).^2 + abs(otfFy).^2;

if D > 1
    fft_double_grad    = repmat(fft_double_grad,[1,1,D]);
    fft_double_laplace = repmat(fft_double_laplace,[1,1,D]);
    weight_x = repmat(weight_x,[1,1,D]);
    weight_y = repmat(weight_y,[1,1,D]);
end

F = 0;
N = 0;

Ix = - imfilter(I,f1); Iy = - imfilter(I,f2);
Normin_I = fft2([Ix(:,end,:) - Ix(:,1,:), -diff(Ix,1,2)] + ...
    [Iy(end,:,:) - Iy(1,:,:); -diff(Iy,1,1)]);
Denormin_N = gamma + alpha * fft_double_laplace + beta;
Normin_gI  = fft_double_laplace .* fft2(I);

i = 0;
while true
    i = i+1;
    prev_F = F;
    
    lambda = min(2^(ii + i), 10^5);
    Denormin_F = lambda * fft_double_grad + alpha * fft_double_laplace + beta;
    
    % update q
    qx = - imfilter(F,f1,'circular') - Ix ;
    qy = - imfilter(F,f2,'circular') - Iy ;
    qx = sign(qx) .* max(abs(qx) - weight_x/lambda, 0);
    qy = sign(qy) .* max(abs(qy) - weight_y/lambda, 0);
    
    % compute fog layer (F)
    Normin_q = [qx(:,end,:) - qx(:,1,:), -diff(qx,1,2)] + ...
        [qy(end,:,:) - qy(1,:,:); -diff(qy,1,1)];
    Normin_gN = fft_double_laplace .* fft2(N);
    
    FF = (lambda * (Normin_I + fft2(Normin_q)) + ...
        alpha * (Normin_gI - Normin_gN) + beta * fft2(delta_I - N)) ...
        ./ Denormin_F;
    F  = real(ifft2(FF));
    
    % compute Noise
    Normin_F = fft_double_laplace .* fft2(F);
    B  = fft2(delta_I - F);
    NN = (alpha * (Normin_gI - Normin_F) + beta * B)./Denormin_N;
    N  = real(ifft2(NN));

    if sum(sum(sum(abs(prev_F - F))))/(H*W) < 10^(-1)
        break
    end
end

% normalize F
for c = 1:D
    Ft = F(:,:,c);
    q = numel(Ft);
    for k = 1:500
        m = sum(Ft(Ft<0));
        n = sum(Ft(Ft>1)-1);
        dt =  (m + n)/q;
        if abs(dt) < 1/q
            break;
        end
        Ft = Ft - dt;
    end
    F(:,:,c) = Ft;
end
F = abs(F);
F(F>1) = 1;

N(N>1) = 1;
N(N<0) = 0;
N = mean(N,3);
G = abs(I - F - N);
G = min(G, [], 3);
G = imgaussfilt(G,3);
F = abs(I - G - N);
F(F == 0) = 0.001;

end


function [weight_x, weight_y] = Gradient_weight(I)
%eps = 10;
I = rgb2gray(I);
lambda = 10;
f1 = [1, -1];
f2 = [1; -1];

Gx = - imfilter(I,f1,'circular');
Gy = - imfilter(I,f2,'circular');

ax = exp(-lambda*abs(Gx));
thx = Gx < 0.01;
ax(thx)=0;
weight_x = 1 + ax;
% weight_x(thx) = 1;
ay = exp(-lambda*abs(Gy));
thy = Gy < 0.01;
ay(thy)=0;
weight_y = 1 + ay;

end

function [Gm] = Compensation ( fog_free_layer,G)
JBack = abs(Background(fog_free_layer));
JBack(JBack > 1)= 1;
JBack(JBack == 0)= 0.001;
GR = G.*fog_free_layer;
Gm = GR./JBack;
end

function [JBack] = Background(RGB)
[height, width, ~] = size(RGB);
patchSize = 3; %the patch size is set to be 15 x 15
padSize = 1; % half the patch size to pad the image with for the array to 
%work (be centered at 1,1 as well as at height,1 and 1,width and height,width etc)
JBack = zeros(height, width); % the dark channel
imJ = padarray(RGB, [padSize padSize], Inf); % the new image
% imagesc(imJ); colormap gray; axis off image

for j = 1:height
    for i = 1:width        
        % the patch has top left corner at (jj, ii)
        patch = imJ(j:(j+patchSize-1), i:(i+patchSize-1),:);
        % the bright channel for a patch is the maximum value for all
        % channels for that patch
        JBack(j,i) = max(patch(:));
    end
end
end


