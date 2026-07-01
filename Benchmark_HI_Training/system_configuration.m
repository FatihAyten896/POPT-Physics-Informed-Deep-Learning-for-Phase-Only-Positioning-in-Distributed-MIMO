function params = system_configuration
    % flags
    params.GROUND_REFLECTION =  1;          % have ground reflection
    params.FIXED_HEIGHT =       1;          % fix UE height
    params.PLOT =               0;          % illustrate EGS
    params.PRINT =              1;          % print performance metrics 
    params.MEX =                1;          % use mex-implementation to compute EGS
    params.DOUBLE_PRECISION =   1;          % use 1 for double-precision and 0 for single-precision
    params.LOOKUP =             1;          % use lookup table
    params.EFIM =               0;

    % ---- Environment  ----    
    params.seed = randi(1024);              % seed for random number generator
    params.z_true = 2;                      % true UE height (only used if FIXED_HEIGHT is true)
    params.ap_height = 4;                   % Access point heights
    params.side_length = 10;                % Side length
    params.delta = .1;                      % Grid size
    
    % ---- Parameters  ----
    params.fc = 800e6;                      % Carrier frequency
    params.c = physconst('LightSpeed');     % Light speed
    params.Ns              = 1;             % Number of subcarriers = 1 (narrowband)
    params.Delta_f         = 180e-6;        % GHz
    params.N0_dBmHz        = -174;          % Noise PSD
    params.NF_dB           = 13;             % Noise figure
    params.P_dBm           = 1;             % Transmit power in dBm
    
    % ---- channel  ----
    params.lambda = params.c/params.fc;
    params.omega = 2*pi/params.lambda;
    params.W = params.Ns * params.Delta_f;                           % GHz
    params.N0 = 10.^((params.N0_dBmHz + params.NF_dB)*0.1) * 1e9;    % W/GHz
    params.sigma2 = params.N0 * params.W;                            % W
    params.A = params.lambda / (4*pi);
    params.P = 10.^(0.1 * params.P_dBm);
    params.SNR_coeff = (4*pi / params.lambda)^2 * params.sigma2 / (2*params.P);

    % ---- Ground reflection  ----
    params.REFLECTION_MODE = 'realistic';
    params.MATERIAL        = 'concrete';
    params.POLARIZATION    = 'TE';

    % ---- AP/VA locations ----
    ap_locs = [ ...
        5.00 5.00;
        4.5  3.0 ;
        2.0  3.73;
        6.5  3.0 ;
        3.0  6.0 ;
        5.0  7.0 ;
        8.0  6.0 ;
        8.0  4.0 ;
        7.0  7.0 ;
        1.2  3.0 ;
        2.3  1.73;
        7.5  1.0 ;
        8.0  2.15;
        1.3  7.0 ;
        0.5  8.9 ;
        8.0  9.2 ;
        7.0  7.9 ]';
    
    ap_locs = vertcat(ap_locs,params.ap_height*ones(1,size(ap_locs,2)));

    va_locs = ap_locs;
    va_locs(3,:) = -va_locs(3,:);

    params.ap_locs = ap_locs;
    params.va_locs = va_locs;
end