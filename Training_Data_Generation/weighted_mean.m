function phi = weighted_mean(x,sigma2)

    if nargin == 1
        phi = mean(x);
    else
        if size(x,2) == 1
            phi = sum(x./sigma2)/sum(1./sigma2);
        else
            phi = sum(x./sigma2,1) ./ sum(1./sigma2,1);
        end
    end

end


% weighted_mean = @(x,sigma2) );
