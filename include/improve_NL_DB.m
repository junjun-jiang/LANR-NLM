function [output] = improve_NL_DB(im_l, im_h, im_SR, upscale, psf, maxIter, conf)
im_SR_org = im_SR;


im_SR = im_SR(upscale+1:end-upscale,upscale+1:end-upscale);
im_l  = im_l(2:end-1,2:end-1);
im_h  = im_h(upscale+1:end-upscale,upscale+1:end-upscale);

fprintf( 'Preprocessing, Iter %d : PSNR = %f; SSIM = %f\n', 0, psnr2(255*im_SR,255*im_h),ssim2(255*im_SR,255*im_h) );

[nrow ncol] = size(im_SR);


f = im_SR(:);               
B   = Set_blur_matrix(im_l, upscale, psf);
BTY = B'*im_l(:);
BTB = B'*B;
flag = 0;

for iter = 1:maxIter

    f_pre = f;

    if mod(iter,40)==0
        %%% here we can updta the im_SR, and obtain the new NLM Matrix                    
        N            =   Compute_NLM_Matrix(reshape(f_pre,[nrow ncol]), 5);
        NTN          =   N'*N*0.006; 
        flag = 1;
    end

    f = f_pre(:);

    for jj = 1:10
        f   = f + 5.*(BTY - BTB*f);                    
    end 
    
    if flag == 1
        f = f - NTN*f_pre(:);  
    end

    if mod(iter,10)==0
        PSNR     =  psnr2(255*reshape(f,[nrow ncol]),255*im_h);
        SSIM     =  ssim2(255*reshape(f,[nrow ncol]),255*im_h);
        fprintf( 'Preprocessing, Iter %d : PSNR = %f; SSIM = %f\n', iter, PSNR,SSIM);
        dif       =  mean((f(:)-f_pre(:)).^2);
        if (dif<1e-4 && iter>=400) 
            break; 
        end 
%     figure, imshow(reshape(f,[nrow ncol]));        
%     im_SRc = cell(1,1);
%     im_SRc{1,1} = reshape(f,size(im_SR));
%     [result, midres] = Update_LANR(conf, im_SRc);
%     figure, imshow(result);
%     PSNR     =  psnr2(255*result,255*im_h)
    end 
    
    flag = 0;
end
im_SR = reshape(f,[nrow ncol]);
im_SR_org(upscale+1:end-upscale,upscale+1:end-upscale) = im_SR;
output = cell(1);
output{1}=im_SR_org;