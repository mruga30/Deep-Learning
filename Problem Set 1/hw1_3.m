%read all the images, parse them and add them to an array of 154*1600
C = []; y = [];

myFolder = 'C:\Users\Owner\Documents\MATLAB\images';
filePattern = fullfile(myFolder, 'subject*.*');
jpegFiles = dir(filePattern);
for k = 1:length(jpegFiles)
  baseFileName = jpegFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  imageArray = double(imread(fullFileName));
  A = imresize(imageArray,'OutputSize',[40 40]);
  B = reshape(A,[1,1600]);
  C = [C;B];
  %create the target matrix with 14 columns according to the subject number
  if baseFileName(8:9) == '02'
      y = [y;1,0,0,0,0,0,0,0,0,0,0,0,0,0];
  elseif baseFileName(8:9) == '03'
      y = [y;0,1,0,0,0,0,0,0,0,0,0,0,0,0];
  elseif baseFileName(8:9) == '04'
      y = [y;0,0,1,0,0,0,0,0,0,0,0,0,0,0];
  elseif baseFileName(8:9) == '05'
      y = [y;0,0,0,1,0,0,0,0,0,0,0,0,0,0];
  elseif baseFileName(8:9) == '06'
      y = [y;0,0,0,0,1,0,0,0,0,0,0,0,0,0];
  elseif baseFileName(8:9) == '07'
      y = [y;0,0,0,0,0,1,0,0,0,0,0,0,0,0];
  elseif baseFileName(8:9) == '08'
      y = [y;0,0,0,0,0,0,1,0,0,0,0,0,0,0];
  elseif baseFileName(8:9) == '09'
      y = [y;0,0,0,0,0,0,0,1,0,0,0,0,0,0];
  elseif baseFileName(8:9) == '10'
      y = [y;0,0,0,0,0,0,0,0,1,0,0,0,0,0];
  elseif baseFileName(8:9) == '11'
      y = [y;0,0,0,0,0,0,0,0,0,1,0,0,0,0];
  elseif baseFileName(8:9) == '12'
      y = [y;0,0,0,0,0,0,0,0,0,0,1,0,0,0];
  elseif baseFileName(8:9) == '13'
      y = [y;0,0,0,0,0,0,0,0,0,0,0,1,0,0];
  elseif baseFileName(8:9) == '14'
      y = [y;0,0,0,0,0,0,0,0,0,0,0,0,1,0];
  elseif baseFileName(8:9) == '15'
      y = [y;0,0,0,0,0,0,0,0,0,0,0,0,0,1];
  end
end

%randomise the data and divide into training and testing sets
data = [C,y];
dtr=[]; dts=[];
row = 11;
rng(1);
ind = randperm(row);
for n = 1:11:153
    c = data(n:n+10,:);
    cn = c(ind,:);
    dtr = [dtr;cn(1:7,:)];
    dts = [dts;cn(8:11,:)];
end

tr = dtr(:,1:1600);
y_tr = dtr(:,1601:1614);
ts = dts(:,1:1600);
y_ts = dts(:,1601:1614);

% Standardise the training data
M = mean(tr); S = std(tr);
tr1 = tr;

for k = 1:size(tr,2)
    for j = 1:size(tr,1)
         tr1(j,k)= tr(j,k)-M(1,k);
         if S(1,k)==0
             tr1(j,k)= 0;
         else
             tr1(j,k)= tr1(j,k)/S(1,k);
         end
    end
end

%Standardising the testing data
for k = 1:size(ts,2)
    for j = 1:size(ts,1)
         ts1(j,k)= ts(j,k)-M(1,k);
         if S(1,k)==0
             ts1(j,k) = 0;
         else
             ts1(j,k)= ts1(j,k)/S(1,k);
         end
    end
end

%add bias
bs = ones(size(tr1,1),1);
tr2 = [bs,tr1];
bs = ones(size(ts1,1),1);
ts2 = [bs,ts1];

%initialize hyperparameters: lr(learning rate), theta
th = zeros(size(tr2,2),size(y_tr,2));
lr = 0.7; th1 = th; alpha = 0;

%calculate y_hat for training and testing data
for m=1:size(y_tr,2)
    for n=1:size(y_tr,1)
        y_hat_dn = 1+exp(-tr2(n,:)*th1(:,m));
        y_hat(n,m) = 1/y_hat_dn;
    end
end

for m=1:size(y_ts,2)
    for n=1:size(y_ts,1)
        y_hats_dn = 1+exp(-ts2(n,:)*th1(:,m));
        y_hats(n,m) = 1/y_hats_dn;
    end
end

%calculate log likelihood for training and testing data
iter=0; ll=[]; it=[]; lrm =[]; lls = [];
one1 = ones(size(y_tr,1),1);
one2 = ones(size(y_ts,1),1);
ll_old = sum(sum(y_tr.*log(y_hat)+(one1-y_tr).*log(one1-y_hat)));
ll_olds = sum(sum(y_ts.*log(y_hats)+(one2-y_ts).*log(one2-y_hats)));
ll_new = ll_old+11+exp(1);
ll_new_ts = ll_olds;

%train the system
while abs(ll_old-ll_new) > 0.001*exp(1) %iter < 140 
    grad_desc = ((transpose(tr2)*(y_tr-y_hat))/size(y_tr,1)) + 2*alpha*th1;
    th1 = th1+(lr*grad_desc);
    if ll_old > ll_new
        lr = 0.94*lr;
    end
    
    % y_hat to calculate log likelihood for training data 
    for m=1:size(y_tr,2)
        for n=1:size(y_tr,1)
            y_hat_dn = 1+exp(-tr2(n,:)*th1(:,m));
            y_hat(n,m) = 1/y_hat_dn;
            if y_hat(n,m) == 0
                y_hat(n,m) = 0.00001;
            elseif y_hat(n,m)== 1
                y_hat(n,m) = 0.99999;
            else
            end
        end
    end
    
    % y_hat to calculate log likelihood for testing data per iteration
    for m=1:size(y_ts,2)
        for n=1:size(y_ts,1)
            y_hats_dn = 1+exp(-ts2(n,:)*th1(:,m));
            y_hats(n,m) = 1/y_hats_dn;
            if y_hats(n,m) == 0
                y_hats(n,m) = 0.00001;
            elseif y_hats(n,m)== 1
                y_hats(n,m) = 0.99999;
            else
            end
        end
    end
    iter = iter+1;
    it = [it,iter];
    lrm = [lrm,lr];
    ll= [ll,ll_old];
    ll_old = ll_new;
    ll_new = sum(sum(y_tr.*log(y_hat)+(one1-y_tr).*log(one1-y_hat)));
    ll_olds = ll_new_ts;
    lls = [lls,ll_olds];
    ll_new_ts = sum(sum(y_ts.*log(y_hats)+(one2-y_ts).*log(one2-y_hats))); % log likelihood of testing data
end

%test the system with the Y calculated 
for m=1:size(y_ts,2)
    for n=1:size(y_ts,1)
        y_calc_dn = 1+exp(-ts2(n,:)*th1(:,m));
        y_calc(n,m) = 1/y_calc_dn;
    end
end

for m=1:size(y_tr,2)
    for n=1:size(y_tr,1)
        y_calcr_dn = 1+exp(-tr2(n,:)*th1(:,m));
        y_calcr(n,m) = 1/y_calcr_dn;
    end
end

%set thresholds
indices = find(abs(y_calcr)<1);
y_calcr(indices) = 0;
indices = find(abs(y_calc)<1);
y_calc(indices) = 0;

%find the accuracy
vtrue =0;
for n = 1:size(y_ts,1)
    if isequal(zeros(1,size(y_calc,2)),y_calc(n,:))
        y_calc(n,1)=1;
    end
    if isequal(y_ts(n,:),y_calc(n,:))
        vtrue=vtrue+1;
    end
end

v1true =0;
for n = 1:size(y_tr,1)
    if isequal(y_tr(n,:),y_calcr(n,:))
        v1true=v1true+1;
    end
end

accuracy_test = (vtrue)*100/size(y_ts,1)
accuracy_training = (v1true)*100/size(y_tr,1)

%confusion matrix
cf = zeros(size(y_tr,2),size(y_tr,2));
for n = 1:size(y_ts,1)
    in1 = find(y_calc(n,:)==1);
    in2 = find(y_ts(n,:)==1);
    cf(in1,in2) = cf(in1,in2)+1;
end
confusionchart(cf)

%create the plot for log likelihood
plot(it,ll,it,lls)
xlabel('Iterations')
ylabel('Log Likelihood')
legend({'Training data','Testing Data'},'Location','southeast')