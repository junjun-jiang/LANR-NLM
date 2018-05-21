function  B   =  Set_blur_matrix(Img, scale, psf)

% scale = 3;
% Img = rand(120,100);
% psf                =     fspecial('gauss', 7, 1.6);              % The simulated PSF

[lh lw ch] =   size(Img);
hh         =   lh*scale;
hw         =   lw*scale;
M          =   lh*lw;
N          =   hh*hw;

ws         =   size(psf, 1 );
t          =   (ws-1)/2;
cen        =   ceil(ws/2);

nv         =   ws*ws;
nt         =   (nv)*M;
R          =   zeros(nt,1);
C          =   zeros(nt,1);
V          =   zeros(nt,1);
cnt        =   1;

pos     =  (1:hh*hw);
pos     =  reshape(pos, [hh hw]);

for lrow = 1:lh
    for lcol = 1:lw
        
        row        =   (lrow-1)*scale + 1;
        col        =   (lcol-1)*scale + 1;
        
        row_idx    =   (lcol-1)*lh + lrow;
        
        
        rmin       =  max( row-t, 1);
        rmax       =  min( row+t, hh);
        cmin       =  max( col-t, 1);
        cmax       =  min( col+t, hw);
        sup        =  pos(rmin:rmax, cmin:cmax);
        col_ind    =  sup(:);
        
        r1         =  row-rmin;
        r2         =  rmax-row;
        c1         =  col-cmin;
        c2         =  cmax-col;
        
        psf2       =  psf(cen-r1:cen+r2, cen-c1:cen+c2);
        psf2       =  psf2(:);

        nn         =  size(col_ind,1);
        
        R(cnt:cnt+nn-1)  =  row_idx;
        C(cnt:cnt+nn-1)  =  col_ind;
        V(cnt:cnt+nn-1)  =  psf2/sum(psf2);
        
        cnt              =  cnt + nn;
    end
end

R   =  R(1:cnt-1);
C   =  C(1:cnt-1);
V   =  V(1:cnt-1);
B   =  sparse(R, C, V, M, N);