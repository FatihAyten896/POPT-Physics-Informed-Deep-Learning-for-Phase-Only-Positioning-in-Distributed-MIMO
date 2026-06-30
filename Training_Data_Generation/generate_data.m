function obj = generate_data(N,params)
    % Uniformly distributed pseudorandom numbers between [a, b].
    randu = @(a,b,N) a + (b-a).*rand(size(a,1),N);

    dx = 0.5;
    lb = [0; 0; 0] + dx;
    ub = [params.side_length; params.side_length; params.ap_height] - dx;
    phi = params.phi;

    % generate training data
    X = randu(lb,ub,N);
    if params.FIXED_HEIGHT, X(3,:) = params.z_true; end

    M = size(params.ap_locs,2);
    Alpha = zeros(M, N);
    Theta = zeros(M, N);
    Sigma = zeros(M, N);
    r = zeros(M, N);
    y = zeros(M, N);
    d_LoS = zeros(M, N);
    d_NLoS = zeros(M, N);
    incidence_angle = zeros(M, N);

    for i = 1:N
        idx = find(vecnorm(params.ap_locs - X(:,i))==0);
        if any(idx)
            X(:,i) = X(:,i) + 1e-2*randn(3,1);
        end

        [alpha,theta,d_los,d_nlos] = ChannelClass.h_func_phasor(X(:,i),params);
        sigma2 = params.sigma2 ./ (2 * params.P .* alpha.^2);

        h = alpha.*exp(-1j.*theta);
        [r_tp,y_tp] = ChannelClass.add_noise(h,phi,params);

        Alpha(:,i) = alpha;
        Theta(:,i) = theta;
        Sigma(:,i) = sigma2;
        r(:,i) = r_tp;
        y(:,i) = y_tp;
        d_LoS(:,i) = d_los;
        d_NLoS(:,i) = d_nlos;

        v = params.va_locs - X(:,i);
        incidence_angle(:,i) = acos(abs(v(3,:)') ./ d_nlos);
    end

    obj.X = X;
    obj.r = r(:);
    obj.y = y(:);
    obj.Alpha = Alpha(:);
    obj.Theta = Theta(:);
    obj.Sigma = Sigma(:);
    obj.d_LoS = d_LoS(:);
    obj.d_NLoS = d_NLoS(:);
    obj.incidence_angle = incidence_angle(:);
    obj.regressor = [obj.d_NLoS-obj.d_LoS obj.incidence_angle];
    obj.r_los = wrap_around(2*pi*(obj.d_LoS/params.lambda - floor(obj.d_LoS/params.lambda)),-pi,pi);
end