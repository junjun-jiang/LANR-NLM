function psnr = psnr2(frame1,frame2)

width = size(frame1,2);
height = size(frame2,1);
diff = abs(double(frame1) - double(frame2));
ssd = sum(sum(diff.*diff));
psnr = 10*log10(width*height*255^2/ssd);
 

