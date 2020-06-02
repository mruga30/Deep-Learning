rng(1);

% Synthetic data where Iv=image with vertical white line, Ih = Image with horizontal white line
Iv = zeros(40,40); Iv(:,5)=1;
Ih = zeros(40,40); Ih(30,:)=1;
yv = 0;
yh = 1;

%Convolution
for i =1:4
    %initialize K and calculate feature maps
    K{i} = zeros(5,5); %imshow(K)
    Fv{i} = conv2(Iv,K{i},'valid');
    Fh{i} = conv2(Ih,K{i},'valid');
    
    %initialize thetas
    thve{i} = zeros(324,1)+0.5;
    thho{i} = zeros(324,1)+0.5;
end

%Max Pooling width=2, stride=2
for i = 1:4
    xind = 1;
    for x = 1:2:35
        yind = 1;
        for y = 1:2:35           
            Zv{i}(xind,yind) = max(max(Fv{i}(x:x+1,y:y+1)));
            [xvi,yvi] = find(Fv{i}(x:x+1,y:y+1)==Zv{i}(xind,yind));
            locv{xind,yind} = [xvi+x-1,yvi+y-1];
            Zh{i}(xind,yind) = max(max(Fh{i}(x:x+1,y:y+1)));
            [xhi,yhi] = find(Fh{i}(x:x+1,y:y+1)==Zh{i}(xind,yind));
            loch{xind,yind} = [xhi+x-1,yhi+y-1];
            yind = yind+1;
        end
        xind = xind+1;
    end
    lcv{1,i} = locv;
    lch{1,i} = loch;
end

ZV = [reshape(Zv{1},1,[]),reshape(Zv{2},1,[]),reshape(Zv{3},1,[]),reshape(Zv{4},1,[])];
ZH = [reshape(Zh{1},1,[]),reshape(Zh{2},1,[]),reshape(Zh{3},1,[]),reshape(Zh{4},1,[])];

thv = [thve{1};thve{2};thve{3};thve{4}];
thh = [thho{1};thho{2};thho{3};thho{4}];

%Initial Calculations
yv_hat = ZV*thv;
yh_hat = ZH*thh;
Jold = (yv - yv_hat)^2 + (yh - yh_hat)^2;
Jnew = Jold; J = [];
iter = 0; itm = []; 
lr = 0.0001; 

%reshape(select(dF/dK)) calculation and grad_k calculations
    for i = 1:5
        for j = 1:5
            Xv = flip(flip(Iv(5-i+1:40-i+1,5-j+1:40-j+1)));
            Xh = flip(flip(Ih(5-i+1:40-i+1,5-j+1:40-j+1)));

            %locations select for all 4 kernels from X
            for n = 1:4
                for iv = 1:18
                    for jv = 1:18
                        selectv(iv,jv) = Xv(lcv{1,n}{iv,jv}(1,1),lcv{1,n}{iv,jv}(1,2));
                        selecth(iv,jv) = Xh(lch{1,n}{iv,jv}(1,1),lch{1,n}{iv,jv}(1,2));
                    end
                end
                svt{n} = reshape(selectv,1,[]);   
                sht{n} = reshape(selecth,1,[]);
                grad_k{n}(i,j) = sum((thve{n}'*(yv_hat-yv)).*svt{n}) + sum((thho{n}'*(yh_hat-yh)).*sht{n});  
            end  
        end
    end

%Gradient calculations
while iter < 100
    grad_th = ZV'*(yv_hat-yv)+ ZH'*(yh_hat-yh);
    thv = thv - 0.5*lr*grad_th;
    thh = thh - 0.5*lr*grad_th;
    
    thve{1} = thv(1:324,1); thve{2} = thv(325:648,1); thve{3} = thv(649:972,1); thve{4} = thv(973:1296,1);
    thho{1} = thh(1:324,1); thho{2} = thh(325:648,1); thho{3} = thh(649:972,1); thho{4} = thh(973:1296,1);
    
    %recalculating the values after the change in the kernel and theta
    for i =1:4
        K{i} = K{i} - 0.5*lr*grad_k{i};
        Fv{i} = conv2(Iv,K{i},'valid');
        Fh{i} = conv2(Ih,K{i},'valid');
    end
    
    for i = 1:4
        xind = 1;
        for x = 1:2:35
            yind = 1;
            for y = 1:2:35            
                Zv{i}(xind,yind) = max(max(Fv{i}(x:x+1,y:y+1)));
                [xvi,yvi] = find(Fv{i}(x:x+1,y:y+1)==Zv{i}(xind,yind));
                locv{xind,yind} = [xvi+x-1,yvi+y-1];
                Zh{i}(xind,yind) = max(max(Fh{i}(x:x+1,y:y+1)));
                [xhi,yhi] = find(Fh{i}(x:x+1,y:y+1)==Zh{i}(xind,yind));
                loch{xind,yind} = [xhi+x-1,yhi+y-1];
                yind = yind+1;
            end
            xind = xind+1;
        end
    end

    ZV = [reshape(Zv{1},1,[]),reshape(Zv{2},1,[]),reshape(Zv{3},1,[]),reshape(Zv{4},1,[])];
    ZH = [reshape(Zh{1},1,[]),reshape(Zh{2},1,[]),reshape(Zh{3},1,[]),reshape(Zh{4},1,[])];
    
    yv_hat = ZV*thv;
    yh_hat = ZH*thh;
    
    %reshape(select(dF/dK)) calculation and grad_k calculations
    for i = 1:5
        for j = 1:5
            Xv = flip(flip(Iv(5-i+1:40-i+1,5-j+1:40-j+1)));
            Xh = flip(flip(Ih(5-i+1:40-i+1,5-j+1:40-j+1)));

            %locations select for all 4 kernels from X
            for n = 1:4
                for iv = 1:18
                    for jv = 1:18
                        selectv(iv,jv) = Xv(lcv{1,n}{iv,jv}(1,1),lcv{1,n}{iv,jv}(1,2));
                        selecth(iv,jv) = Xh(lch{1,n}{iv,jv}(1,1),lch{1,n}{iv,jv}(1,2));
                    end
                end
                svt{n} = reshape(selectv,1,[]);   
                sht{n} = reshape(selecth,1,[]);
                grad_k{n}(i,j) = sum((thve{n}'*(yv_hat-yv)).*svt{n}) + sum((thho{n}'*(yh_hat-yh)).*sht{n});  
            end  
        end
    end
    
    if Jnew > Jold
       lr = 0.1*lr; 
    end
    
    % store values so as to plot later
    itm = [itm,iter];
    J = [J,Jold];
    
    % calculate LSE
    Jold = Jnew;
    Jnew = (yv - yv_hat)^2 + (yh - yh_hat)^2;
    iter = iter+1;
end

%imshow(K{1},[-0.01,0.01])
%imshow(K{2},[-0.01,0.01])
%imshow(K{3},[-0.01,0.01])
%imshow(K{4},[-0.01,0.01])
plot(itm,J)