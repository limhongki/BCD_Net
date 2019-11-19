function xs = func_minimize_fx_rx(x, Gb, yi, ri, u, ld, niter)


aa = ones(size(x))*2*ld;
aj = Gb' * ones(size(yi));
xs = [];

for i = 1:niter
    x_old = x;
    yp = Gb*x + ri;
    yp(yp==0) = inf;
    ej = Gb' * (yi ./ yp);
    bb = aj - 2*ld*u;
    bb = 1/2*bb;
    cc = -ej.*x;
    cc = -cc;
    
    x = single(real(eql_root(aa,bb,cc)));
    xs = cat(4,xs,x);    
    printf('Iter: %g, Range %g %g, relative diff: %g, %s', i, min(x(:)), max(x(:)), norm(x(:)-x_old(:))/norm(x(:))*100, mfilename)
    im(x(:,:,65));drawnow;
    
end