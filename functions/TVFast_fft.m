function x = TVFast_fft( y1,y2, size_unfold,  lam)
% min ||y1-x||_F^2 + lam*||y2-TV(x)||_F^2
%    y2==[]:  min ||y1-x||_F^2 + lam*||TV(x)||_F^2
% (I + lam D^*D)x=y
sizeD = size_unfold;
L = length(sizeD);
y1 = reshape(y1,sizeD);
if isempty(y2)
    y=y1;
    if L==1
    else if L==2
            Eny_x   = ( abs(psf2otf([+1; -1], sizeD)) ).^2  ;
            Eny_y   = ( abs(psf2otf([+1, -1], sizeD)) ).^2  ;
            determ  =  Eny_x + Eny_y ;
            x        = real( ifftn( fftn(y) ./ (lam*determ + 1) ) );
        else if L==3
                Eny_x   = ( abs(psf2otf([+1; -1], sizeD)) ).^2  ;
                Eny_y   = ( abs(psf2otf([+1, -1], sizeD)) ).^2  ;
                Eny_z   = ( abs(psf2otf([+1, -1], [sizeD(2),sizeD(3),sizeD(1)])) ).^2  ;
                Eny_z   =  permute(Eny_z, [3, 1 2]);
                determ  =  Eny_x + Eny_y + Eny_z;
                x        = real( ifftn( fftn(y) ./ (lam*determ + 1) ) );
            end
        end
    end
else
    if L==1
    else if L==2
            y       = y1 + lam* reshape(diffT2(y2, sizeD),sizeD);
            Eny_x   = ( abs(psf2otf([+1; -1], sizeD)) ).^2  ;
            Eny_y   = ( abs(psf2otf([+1, -1], sizeD)) ).^2  ;
            determ  =  Eny_x + Eny_y ;
            x        = real( ifftn( fftn(y) ./ (lam*determ + 1) ) );
        else if L==3
                y       = y1 + lam* reshape(diffT3(y2, sizeD),sizeD);
                Eny_x   = ( abs(psf2otf([+1; -1], sizeD)) ).^2  ;
                Eny_y   = ( abs(psf2otf([+1, -1], sizeD)) ).^2  ;
                Eny_z   = ( abs(psf2otf([+1, -1], [sizeD(2),sizeD(3),sizeD(1)])) ).^2  ;
                Eny_z   =  permute(Eny_z, [3, 1, 2]);
                determ  =  Eny_x + Eny_y + Eny_z;
                x        = real( ifftn( fftn(y) ./ (lam*determ + 1) ) );
            end
        end
    end
end

