function varargout = TrainAndSample(varargin)
   
    params = system_configuration;
    if nargin > 0
        params.seed = varargin{1};
        params.P = 10.^(0.1 * varargin{2}(1));
        params.N_train           = varargin{2}(2);
        params.FIXED_HEIGHT = varargin{2}(3);
        params.z_true = varargin{2}(4);
        params.MATERIAL        = varargin{3};
        params.POLARIZATION    = varargin{4};
    else
        params.seed = 1;
        params.P = 10.^(0.1 * 0);
        params.N_train = 10; % number of UE locations to train GP model
        params.FIXED_HEIGHT = 0;
        params.z_true = 2;
        params.MATERIAL        = 'concrete';
        params.POLARIZATION    = 'TE';
    end

    maxNumCompThreads(1);

    ADD_RANDOM_CPO = 1;
    TEST_FLAG = 0;

    % number of UE location to generate data for deep learning approach
    N_sample = 1.25e6;
    
    % initialize random seed
    rng(params.seed,'twister')

    params.phi = 2*pi*rand - pi;    


    %% GP training

    % compute permittivity
    params = ChannelClass.complex_permittivity(params);
 
    % generate training and test data
    obj_train = generate_data(params.N_train,params); 

    % Estimate hyperparameters
    obj_train = gp_model.optimize(obj_train, params, true);


    %% generate training data for NN

    % Pre-compute inverse of the covariance function and nu that used in
    % computing the predictive mean and covariance of the posterior
    K = gp_model.kernel(obj_train.regressor,obj_train.regressor,obj_train.theta) + diag(obj_train.Sigma);
    invSqrtK = chol(K,'lower') \ eye(size(obj_train.regressor,1));
    nu = invSqrtK*obj_train.y;

    % Pre-allocate arrays to store data
    N_minibatch = 10; % this is quite close to optimal value
    N = ceil(N_sample/N_minibatch);
    M = size(params.ap_locs,2);
    H = kron(eye(N_minibatch),ones(M,1));

    X = cell(1,N);
    R_theoretical = cell(1,N);
    R_gp = cell(1,N);
    sigma2_gp = cell(1,N);

    warning off
    pw = PoolWaitbar(N, 'Generating data, please wait ...');

    params.phi = 0; % we can set this to zero since we are not using the theoretical model anymore
    parfor i = 1:N
        % initialize random seed
        rng(params.seed + i,'twister')

        % generate data
        obj_test = generate_data(N_minibatch,params);

        % compute predictive mean and covariance of the posterior 
        K_s = gp_model.kernel(obj_train.regressor,obj_test.regressor,obj_train.theta);
        K_ss = gp_model.kernel(obj_test.regressor,obj_test.regressor,obj_train.theta);
        v = invSqrtK*K_s;
        E = K_s'*invSqrtK'*nu;
        V = K_ss - v'*v;
        
        % compute moments
        if ADD_RANDOM_CPO
            % generate random CPO for every UE location
            phi = 2*pi*rand(N_minibatch,1) - pi;
            mu = wrap_around(obj_test.r_los + H*phi + E,-pi,pi);
        else
            mu = wrap_around(obj_test.r_los + E,-pi,pi);
        end
        sigma2 = diag(V) + obj_test.Sigma;

        % sample and handle wrap-around
        r_gp = mu + sqrt(sigma2) .* randn(M*N_minibatch,1);
        r_gp = wrap_around(r_gp,-pi,pi);

        % store measurements
        X{i} = obj_test.X;
        R_theoretical{i} = reshape(obj_test.r,M,N_minibatch);
        R_gp{i} =  reshape(r_gp,M,N_minibatch);
        %sigma2_gp{i} = reshape(sigma2,M,N_minibatch);

        if TEST_FLAG
            % sanity check that the compute mean and covariance are the
            % same as given "gp_model2.posterior_predictive"
            [E_,V_] = gp_model.posterior_predictive(obj_test.regressor, ...
                    obj_train.regressor, obj_train.y, obj_train.Sigma, obj_train.theta);
            assert(sum((E - E_).^2,'all') < 1e-12)
            assert(sum((V - V_).^2,'all') < 1e-12)

            drawnow
        end
        increment(pw);
    end
    delete(pw);
    warning on

    if nargin > 0
        varargout{1} = cell2mat(X);
        varargout{2} = cell2mat(R_theoretical);
        varargout{3} = cell2mat(R_gp);
        %varargout{4} = cell2mat(sigma2_gp);
    end
end

%clc;clear all; close all;
n_gp_values = [5,10,50];       % GP training points

p_t_values = -50:10:0;        % Transmit power values

baseFolder = 'C:\Users\ksb394\OneDrive - TUNI.fi\Desktop\TWC Journal\Reproducibility\Data';

for i = 1:length(n_gp_values)
    n_gp = n_gp_values(i);
    for j = 1:length(p_t_values)
        p_t = p_t_values(j);
        fprintf('Running for n_gp = %d, p_t = %d dBm\n', n_gp, p_t);
        [X, Y_theoretical, Y_gp] = TrainAndSample(1, [p_t n_gp 0 0], 'concrete', 'TE');
        outFolder = fullfile(baseFolder, sprintf('%ddBm', p_t));
        if ~exist(outFolder, 'dir')
            mkdir(outFolder);
        end
        save(fullfile(outFolder, sprintf('X_ngp_%d_%d_dbm.mat', n_gp, p_t)), 'X');
        save(fullfile(outFolder, sprintf('Y_theoretical_ngp_%d_%d_dbm.mat', n_gp, p_t)), 'Y_theoretical');
        save(fullfile(outFolder, sprintf('Y_gp_ngp_%d_%d_dbm.mat', n_gp, p_t)), 'Y_gp');

    end
end