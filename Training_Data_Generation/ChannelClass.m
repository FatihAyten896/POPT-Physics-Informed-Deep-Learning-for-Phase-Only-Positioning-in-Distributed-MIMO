classdef ChannelClass
    methods (Static)
        function [h,d_LoS] = h_func(x,params)
            % get parameters
            ap_locs = params.ap_locs;
            va_locs = params.va_locs;
            A = params.A;
            omega = params.omega;

            % LoS channel
            d_LoS = sqrt(sum((ap_locs - x).^2,1))';
            h = A .* exp(-1j * omega .* d_LoS) ./ d_LoS;

            % NLoS channel (ground reflection)
            if params.GROUND_REFLECTION
                % compute incidence angles
                v = va_locs - x;
                d_NLoS = sqrt(sum(v.^2,1))';
                incidence_angle = acos(abs(v(3,:)') ./ d_NLoS);

                % compute reflection coefficient
                Gammas = ChannelClass.reflection_coefficient(incidence_angle,params);

                % NLoS channel
                h_nlos  = A .* exp(-1j * omega .* d_NLoS ) ./ d_NLoS;
                h  = h + Gammas .* h_nlos;
            end
        end

        function [alpha,theta,d_LoS,d_NLoS] = h_func_phasor(x,params)
            % get parameters
            ap_locs = params.ap_locs;
            va_locs = params.va_locs;
            A = params.A;
            omega = params.omega;

            % LoS channel
            d_LoS = sqrt(sum((ap_locs - x).^2,1))';
            alpha_los = A./ d_LoS;
            theta_los = omega .* d_LoS;

            % NLoS channel (ground reflection)
            if params.GROUND_REFLECTION
                % compute incidence angles
                v = va_locs - x;
                d_NLoS = sqrt(sum(v.^2,1))';
                incidence_angle = acos(abs(v(3,:)') ./ d_NLoS);
                Gamma = ChannelClass.reflection_coefficient(incidence_angle,params);

                h_real = real(Gamma);
                h_imag = imag(Gamma);

                phase_gamma = atan2(h_imag,h_real);
                amplitude_gamma = sqrt(h_real.^2 + h_imag.^2);

                alpha_nlos =  amplitude_gamma .* A./ d_NLoS;
                theta_nlos = omega .* d_NLoS - phase_gamma;

                alpha = sqrt(alpha_los.^2+alpha_nlos.^2+2*alpha_los.*alpha_nlos.*cos(theta_los-theta_nlos));
                a = alpha_los.*sin(theta_los)+alpha_nlos.*sin(theta_nlos);
                b = alpha_los.*cos(theta_los)+alpha_nlos.*cos(theta_nlos);
                theta = atan2(a,b);
            else
                disp('error')
                alpha = alpha_los;
                theta = theta_los;

                d_NLoS = [];
            end
        end

        function [r_tp, y_tp] = add_noise(h,phi,params)
            % get parameters
            sigma2 = params.sigma2;
            P = params.P;
            n = size(h,1);

            % common UE network random phase (per snapshot)
            cpo = exp(-1j*phi);

            % noise & received signal
            std_cplx = sqrt(sigma2/2);
            n_tp = std_cplx * (randn(n,1) + 1j*randn(n,1));
            y_tp = sqrt(P) .* (h .* cpo) + n_tp;

            % noisy phase measurement
            r_tp = -angle(y_tp);
        end

        function Sigma = compute_variance(x,params)
            [alpha,theta] = ChannelClass.h_func_phasor(x,params);

            % compute measurement covariance
            h = sqrt(params.P)*alpha.*exp(-1i*theta);
            H = [-1i*h  diag(h./alpha) diag(-1i*h)];
            FIM = 2./params.sigma2 .* real(H'*H);

            M = size(params.ap_locs,2);
            idx_nuisance = 1:M+1;
            idx_interest = setdiff(1:2*M+1,idx_nuisance);

            if params.EFIM
                % FIM = 2./params.sigma2 .* real(H'*H);
                A = FIM(idx_interest,idx_interest);
                B = FIM(idx_nuisance,idx_nuisance);
                C = FIM(idx_nuisance,idx_interest);
                EFIM = A - C'*pinv(B)*C;
                Sigma = (1./diag(EFIM));
            else
                Sigma = (1./diag(FIM(idx_interest,idx_interest)));
            end
        end

        function Gammas = reflection_coefficient(theta,params)
            if strcmpi(params.REFLECTION_MODE, 'simple')
                Gammas = ones(size(theta));
            elseif strcmpi(params.MATERIAL, 'metal')
                Gammas = -ones(size(theta));
            else
                eps_rc = params.eps_rc;
                cos_t = cos(theta);
                sin_t = sin(theta);
                root = sqrt(eps_rc - (sin_t.^2));
                
                if strcmpi(params.POLARIZATION,'TE')
                    Gammas = (cos_t - root) ./ (cos_t + root);
                else
                    Gammas = (eps_rc .* cos_t - root) ./ (eps_rc .* cos_t + root);
                end
            end
        end

        function params = complex_permittivity(params)
            f_GHz = params.fc;
            material_key = params.MATERIAL;

            switch lower(material_key)
                case 'concrete',     a=5.24; b=0.0; cval=0.0462; d=0.7822;
                case 'glass',        a=6.31; b=0.0; cval=0.0036; d=1.3394;
                case 'plasterboard', a=2.73; b=0.0; cval=0.0085; d=0.9395;
                case 'wood',         a=1.99; b=0.0; cval=0.0047; d=1.0718;
                case 'metal', params.eps_rc = complex(1e6,-1e6); return;

                otherwise, error("Unknown material '%s'.", material_key);
            end
            eps_r_prime = a * (f_GHz.^b);
            sigma = cval * (f_GHz.^d);
            eps_r_dblp = 17.98 * sigma ./ f_GHz;
            params.eps_rc = complex(eps_r_prime, -eps_r_dblp);
        end
    end
end
