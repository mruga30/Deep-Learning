rng(0);

% Synthetic data where Iv=image with vertical white line, Ih = Image with horizontal white line
Iv = zeros(40,40); Iv(:,5)=1;
Ih = zeros(40,40); Ih(30,:)=1;
K = zeros(40,40);
%imshow(K)
thv = randn(1,1);
thh = randn(1,1);
yv = 0;
yh = 1;
Fv = 0; Fh = 0;

% Convolution
Fv = sum(sum(Iv.*flip(flip(K,1),2)));
Fh = sum(sum(Ih.*flip(flip(K,1),2)));

% No pooling is done because the pool layer has width=1,stride=1 that is equal to the image

%Initial Calculations
yv_hat = 1./(1+exp(-Fv.*thv));
yh_hat = 1./(1+exp(-Fh.*thh));
Jold = yv*log(yv_hat)+(1-yv).*log(1-yv_hat)+yh*log(yh_hat)+(1-yh).*log(1-yh_hat);
Jnew = Jold; J = [];
iter = 0; itm = []; 
lr = 0.09;

% Gradient calculations
while iter <100 
    grad_th = Fv*(yv_hat-yv)+ Fh*(yh_hat-yh);
    grad_k = thv*(yv_hat-yv)*flip(flip(Iv,1),2) + thh*(yh_hat-yh)*flip(flip(Ih,1),2); %we dont do select/reeshape here because the
    thv = thv + 0.5*lr*grad_th;
    thh = thh + 0.5*lr*grad_th;
    K = K + 0.5*lr*grad_k;
    
    %recalculating the values after the change in the kernel and theta
    Fv = sum(sum(Iv.*flip(flip(K,1),2)));
    Fh = sum(sum(Ih.*flip(flip(K,1),2)));
    yv_hat = 1./(1+exp(-Fv.*thv));
    yh_hat = 1./(1+exp(-Fh.*thh));
    
    %avoid Nan/Inf
    if yv_hat == 0
        yv_hat = 0.0001;
    end
    
    if yh_hat == 0
        yh_hat = 0.0001;
    end
    
    if yv_hat == 1
        yv_hat = 0.9999;
    end
    
    if yh_hat == 1
        yh_hat = 0.9999;
    end
    
    if Jnew > Jold
       lr = 0.5*lr; 
    end
    
    %store values so as to plot later
    itm = [itm,iter];
    J = [J,Jold];
    
    %calculate Log likelihood    
    Jold = Jnew;
    Jnew = yv*log(yv_hat)+(1-yv).*log(1-yv_hat)+yh*log(yh_hat)+(1-yh).*log(1-yh_hat);
    iter = iter+1;
end

%imshow(K,[-10^8,10^8])
plot(itm,J)