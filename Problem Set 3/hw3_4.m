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
yv_hat = exp(Fv*thv)/(exp(Fv*thv)+exp(Fh*thh));
yh_hat = exp(Fh*thh)/(exp(Fv*thv)+exp(Fh*thh));
Jold = -log(yv_hat)-log(yh_hat);
Jnew = Jold;
iter = 0; itm = []; 
lr = 1;

% Gradient calculations
while iter <100    
    grad_th = Fv*(yv_hat-yv)+ Fh*(yh_hat-yh);
    grad_k = thv*(yv_hat-yv)*flip(flip(Iv,1),2) + thh*(yh_hat-yh)*flip(flip(Ih,1),2); %we dont do select/reeshape here because the
    thh = thh - 0.5*lr*grad_th;
    thv = thv - 0.5*lr*grad_th;
    K = K - 0.5*lr*grad_k;
    
    %recalculating the values after the change in the kernel and theta
    Fv = sum(sum(Iv.*flip(flip(K,1),2)));
    Fh = sum(sum(Ih.*flip(flip(K,1),2)));
    yv_hat = exp(Fv*thv)/(exp(Fv*thv)+exp(Fh*thh));
    yh_hat = exp(Fh*thh)/(exp(Fv*thv)+exp(Fh*thh));
    
    if Jnew > Jold
       lr = 0.9*lr; 
    end
    
    %store values so as to plot later
    itm = [itm,iter];
    
    %calculate Cross Entropy
    Jold = Jnew;
    Jnew = -log(yv_hat)-log(yh_hat);
    iter = iter+1;
end

%imshow(K,[-0.5,0.5])
plot(itm,J)