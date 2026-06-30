function nu = compute_residual(x,y)
    nu = wrap_around(x - y,-pi,pi);
    idx_pos = nu > pi/2;
    idx_neg = nu < -pi/2;
    pos = sum(idx_pos);
    neg = sum(idx_neg);
    if pos > neg
        nu(idx_neg) = nu(idx_neg) + 2*pi;
    elseif pos < neg
        nu(idx_pos) = nu(idx_pos) - 2*pi;
    end
end
