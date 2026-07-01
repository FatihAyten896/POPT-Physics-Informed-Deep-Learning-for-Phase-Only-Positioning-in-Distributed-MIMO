clc;
clear;
close all;

% Load UE locations
data = load('C:\Users\ksb394\OneDrive - TUNI.fi\Desktop\TWC Journal\Reproducibility\Data\0dBm\X_ngp_5_0_dbm.mat');
ue_locs = data.X;   % 3 x N

% AP locations: 3 x 17
params  = system_configuration;
ap_locs = params.ap_locs;

% Check dimensions
if size(ue_locs,1) ~= 3
    error('ue_locs must have size 3 x N');
end

if size(ap_locs,1) ~= 3 || size(ap_locs,2) ~= 17
    error('ap_locs must have size 3 x 17');
end

% Number of UEs and APs
N   = size(ue_locs, 2);
Nap = size(ap_locs, 2);

% Preallocate distance matrix: 17 x N
dist_matrix = zeros(Nap, N);

% Compute distances between each AP and each UE
for ap = 1:Nap
    diff = ue_locs - ap_locs(:, ap);            
    dist_matrix(ap, :) = sqrt(sum(diff.^2, 1));  
end

% Wavelength for 0.8 GHz
c = 3e8;
f = 0.8e9;
lambda = c / f;

% Integer ambiguities: z1, z2, ..., z17
z_matrix = floor(dist_matrix / lambda);   % 17 x N

% Differential ambiguities w.r.t. the first ambiguity:
% [z2-z1; z3-z1; ...; z17-z1]
diff_ambiguities = z_matrix(2:end, :) - z_matrix(1, :);   % 16 x N

% Save to .mat file
save('C:\Users\ksb394\OneDrive - TUNI.fi\Desktop\TWC Journal\Reproducibility\Benchmark_HI_Training\ambiguities.mat', 'diff_ambiguities');