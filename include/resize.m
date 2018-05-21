function imgs = resize(imgs, scale, method, verbose)

if nargin < 4
    verbose = 0;
end

h = [];
if verbose
    fprintf('Scaling %d images by %.2f (%s) ', numel(imgs), scale, method);
end

for i=1:numel(imgs)
    h = progress(h, i/numel(imgs), verbose);
    
    psf         = fspecial('gauss', 7, 1.6);    
    im_l = Blur('fwd', imgs{i}, psf);
    imgs{i} = im_l(1:scale:end,1:scale:end);   
end
if verbose
    fprintf('\n');
end
