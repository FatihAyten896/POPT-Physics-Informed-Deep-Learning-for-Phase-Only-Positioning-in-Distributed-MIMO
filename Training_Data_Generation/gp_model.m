classdef gp_model
    properties (Constant)
        prior = [1 1 1 1]';
    end

    methods (Static)
        function varargout = kernel(X,Y,theta)
            d_se1 = pdist2(X(:,1), Y(:,1),'squaredeuclidean');
            d_se2 = pdist2(X(:,2), Y(:,2),'squaredeuclidean');
            d_pe = pdist2(X(:,1), Y(:,1),'euclidean');

            % squared exponential
            K_se1 = exp(-1/(2*theta(2)) .* d_se1);
            K_se2 = exp(-1/(2*theta(3)) .* d_se2);
            
            % periodic
            lambda = (physconst('LightSpeed')/800e6);
            omega = 2*pi/lambda;
            sin2d = (sin(0.5*omega*d_pe)).^2;
            K_pe = exp(-2/theta(4) .* sin2d);

            % quasi-periodic
            K = K_se1 .* K_se2 .*K_pe;
            varargout{1} = theta(1) .* K;

            if nargout > 1
                varargout{2} = K;
                varargout{3} = theta(1)/(2*theta(2).^2).*d_se1.*K;
                varargout{4} = theta(1)/(2*theta(3).^2).*d_se2.*K;
                varargout{5} = 2*theta(1)/(theta(4).^2).*sin2d.*K;
            end
        end


        function obj = optimize_analytical(obj,params,gradients)
            options = optimset('Display','off','GradObj','off','TolX',1e-9,'TolFun',1e-9);
            if gradients, options.GradObj = 'on'; end

            theta0 = gp_model.prior;
            theta0 = log(theta0);
            X = obj.regressor;

            M = size(params.ap_locs,2);
            N = round(size(obj.r,1)./M);
            
            % use two-ray model to estimate CPO 
            phi = weighted_mean(reshape(compute_residual(obj.r,obj.Theta),M,N),reshape(obj.Sigma,M,N));
            phi = kron(phi',ones(M,1));
            y = compute_residual(obj.r,obj.r_los + phi);

            % Optimize hyperparameters
            fun = @(theta) gp_model.gauss_pdf(X,y,obj.Sigma,theta);
            time_start = tic;
            try
                [theta,fval,exitflag,output] = fminunc(fun,theta0,options);
            catch
                lb = [log([0.001 0.001 0.001 0.001])]';
                ub = [log([100 100 100 100])]';

                % optimization failed, try another optimization algorithm
                [theta,fval,exitflag,output] = fmincon(fun,theta0,[],[],[],[],lb,ub,[],options);
            end
            obj.theta = exp(theta);
            obj.phi = phi;
            obj.y = y;
            obj.dt = toc(time_start);
            obj.fval = fval;
            obj.exitflag = exitflag;
            obj.output = output;
            
            
            % E = gp_model.posterior_predictive(X,X,obj.y,obj.Sigma,obj.theta);
            % nu = compute_residual(obj.r,obj.r_los + obj.phi + E);
            % fprintf("Uniq. CPO (analytical) -- CPO error: %.4f [deg], Training RMSE: %.4f [deg], Cost: %.4f, Exit: %d, Iter: %d, Time: %.4f [s]\n",...
            %     [sqrt(mean((params.phi - obj.phi).^2)) sqrt(mean(nu.^2))]*180/pi, obj.fval, obj.exitflag, obj.output.iterations, obj.dt)
        end


        function obj = optimize(obj, params, gradients)
            options = optimset('Display','off','GradObj','off','TolX',1e-9,'TolFun',1e-9);
            if gradients, options.GradObj = 'on'; end

            X = obj.regressor;
            r = obj.r;
            r_los = obj.r_los;
            Sigma = obj.Sigma;

            M = size(params.ap_locs,2);
            N = round(size(obj.r,1)./M);

            phi0 = weighted_mean(reshape(compute_residual(r,r_los),M,N),reshape(Sigma,M,N));

            theta0 = gp_model.prior;
            theta0 = log(theta0);
            theta0 = [theta0; phi0'];

            % Optimize hyperparameters
            fun = @(theta) gp_model.gauss_pdf_joint(obj,params,theta);
            time_start = tic;
            try
                [theta,fval,exitflag,output] = fminunc(fun,theta0,options);
            catch
                lb = [log([0.001 0.001 0.001 0.001]) -pi ]';
                ub = [log([100 100 100 100]) pi]';

                % optimization failed, try another optimization algorithm
                [theta,fval,exitflag,output] = fmincon(fun,theta0,[],[],[],[],lb,ub,[],options);
            end

            obj.theta = exp(theta(1:4));
            obj.phi = kron(theta(5:end),ones(M,1));
            obj.y = compute_residual(obj.r,obj.r_los + obj.phi);
            obj.dt = toc(time_start);
            obj.fval = fval;
            obj.exitflag = exitflag;
            obj.output = output;
            % 
            % E = gp_model.posterior_predictive(X,X,obj.y,obj.Sigma,obj.theta);
            % nu = compute_residual(obj.r,obj.r_los + obj.phi + E);
            % fprintf("Uniq. CPO (analytical) -- CPO error: %.4f [deg], Training RMSE: %.4f [deg], Cost: %.4f, Exit: %d, Iter: %d, Time: %.4f [s]\n",...
                % [sqrt(mean((params.phi - obj.phi).^2)) sqrt(mean(nu.^2))]*180/pi, obj.fval, obj.exitflag, obj.output.iterations, obj.dt)

        end

        function [E,V] = posterior_predictive(X_,X,y,Sigma,theta)
            N = size(X,1);
            I = eye(N);
            K = gp_model.kernel(X,X,theta) + diag(Sigma);
            K_s = gp_model.kernel(X,X_,theta);
            K_ss = gp_model.kernel(X_,X_,theta);

            sqrtK = chol(K,'lower');
            invSqrtK = sqrtK\I;
            nu = invSqrtK*y;
            v = invSqrtK*K_s;
            E = K_s'*invSqrtK'*nu;
            V = K_ss - v'*v;
        end


        function [cost,gradient] = gauss_pdf(X,y,Sigma,theta)
            % Important! The signs of cost and gradient are opposite since the
            % optimization function finds the local minimum, whereas the
            % derivations imply that we are maximizing the marginal likelihood
            
            theta = exp(theta);

            N = size(X,1);
            if nargout  > 1
                [K,dK_theta1,dK_theta2,dK_theta3,dK_theta4] = gp_model.kernel(X,X,theta);
            else
                K = gp_model.kernel(X,X,theta);
            end

            Ky =  K + diag(Sigma);
            [L,flag] = chol(Ky,'lower');
            if flag ~= 0 
                disp('here')
                L = chol(Ky + 1e-9.*eye(N),'lower'); 
            end

            invL = L\eye(N);
            nu = invL*y;
            cost = 0.5*(nu'*nu + 2*sum(log(diag(L))) + N*log(2*pi));

            if nargout > 1
                gradient = zeros(4,1);
                invKy = invL'*invL;

                gradient(1) = y'*invKy*dK_theta1*invKy*y - trace(invKy*dK_theta1);
                gradient(2) = y'*invKy*dK_theta2*invKy*y - trace(invKy*dK_theta2);
                gradient(3) = y'*invKy*dK_theta3*invKy*y - trace(invKy*dK_theta3);
                gradient(4) = y'*invKy*dK_theta4*invKy*y - trace(invKy*dK_theta4);

                % Account for the log-transformed values
                gradient = theta .* gradient;
                gradient = -0.5*gradient;
            end
        end


        function [cost,gradient] = gauss_pdf_joint(obj,params,theta)
            % Important! The signs of cost and gradient are opposite since the
            % optimization function finds the local minimum, whereas the
            % derivations imply that we are maximizing the marginal likelihood
            
            X = obj.regressor;
            r = obj.r;
            r_los = obj.r_los;
            Sigma = obj.Sigma;

            M = size(params.ap_locs,2);
            N_train = params.N_train;
            H = kron(eye(N_train),ones(M,1));

            phi = theta(5:end);
            theta(1:4) = exp(theta(1:4));

            N = size(X,1);
            if nargout  > 1
                [K,dK_theta1,dK_theta2,dK_theta3,dK_theta4] = gp_model.kernel(X,X,theta(1:4));
            else
                K = gp_model.kernel(X,X,theta(1:4));
            end

            Ky =  K + diag(Sigma);
            [L,flag] = chol(Ky,'lower');
            if flag ~= 0 
                disp('here')
                L = chol(Ky + 1e-9.*eye(N),'lower'); 
            end

            y = compute_residual(r,r_los + H*phi);
            invL = L\eye(N);
            nu = invL*y;
            cost = 0.5*(nu'*nu + 2*sum(log(diag(L))) + N*log(2*pi));

            if nargout > 1
                gradient = zeros(4+N_train,1);
                invKy = invL'*invL;

                gradient(1) = y'*invKy*dK_theta1*invKy*y - trace(invKy*dK_theta1);
                gradient(2) = y'*invKy*dK_theta2*invKy*y - trace(invKy*dK_theta2);
                gradient(3) = y'*invKy*dK_theta3*invKy*y - trace(invKy*dK_theta3);
                gradient(4) = y'*invKy*dK_theta4*invKy*y - trace(invKy*dK_theta4);
                gradient(5:end) = 2*H'*invKy*y;

                % Account for the log-transformed values
                gradient(1:4) = theta(1:4) .* gradient(1:4);
                gradient = -0.5*gradient;
            end
        end
    end
end