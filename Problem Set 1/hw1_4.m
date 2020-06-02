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
sums = 0;
for n=1:size(th1,2)
    sums = sums + exp(tr2*th1(:,n));
end

for m=1:size(y_tr,2)
    for n=1:size(y_tr,1)
        y_hat_nm = exp(tr2(n,:)*th1(:,m));
        y_hat(n,m) = y_hat_nm/sums(n,1);
    end
end

sums_ts = 0;
for n=1:size(th1,2)
    sums_ts = sums_ts + exp(ts2*th1(:,n));
end

for m=1:size(y_ts,2)
    for n=1:size(y_ts,1)
        y_hat_nm_ts = exp(ts2(n,:)*th1(:,m));
        y_hat_ts(n,m) = y_hat_nm_ts/sums_ts(n,1);
    end
end

%calculate cross entropy for training and testing data
iter=0; ce=[]; it=[]; lrm =[]; ces =[];
ce_old = sum(sum(-log(y_hat)));
ce_old_ts = sum(sum(-log(y_hat_ts))); 
ce_new = ce_old+11+exp(1);
ce_new_ts = ce_old_ts;

%train the system
while abs(ce_old-ce_new) > 0.001*exp(1) %iter < 140 
    grad_desc = ((transpose(tr2)*(y_hat-y_tr))/size(y_tr,1)) + 2*alpha*th1;
    th1 = th1-(lr*grad_desc);
    if ce_old < ce_new
        lr = 0.94*lr;
    end
    
    %y_hat for training data (to calculate cross entropy) 
    sums = zeros(size(sums));
    for n=1:size(th1,2)
        sums = sums + exp(tr2*th1(:,n));
    end
    
    for m=1:size(y_tr,2)
        for n=1:size(y_tr,1)
            y_hat_nm = exp(tr2(n,:)*th1(:,m));
            y_hat(n,m) = y_hat_nm/sums(n,1);
        end
    end
    
    %y_hat for testing data (to calculate cross entropy) 
    sums_ts = zeros(size(sums_ts));
    for n=1:size(th1,2)
        sums_ts = sums_ts + exp(ts2*th1(:,n));
    end
    
    for m=1:size(y_ts,2)
        for n=1:size(y_ts,1)
            y_hat_nm_ts = exp(ts2(n,:)*th1(:,m));
            y_hat_ts(n,m) = y_hat_nm_ts/sums_ts(n,1);
        end
    end

    iter = iter+1;
    it = [it,iter];
    lrm = [lrm,lr];
    ce= [ce,ce_old];
    ce_old = ce_new;
    ce_new = sum(sum(-log(y_hat)));
    ces = [ces,ce_old_ts];
    ce_old_ts = ce_new_ts;
    ce_new_ts = sum(sum(-log(y_hat_ts))); %cross entropy of testing data
end

%test the system with the Y calculated 
sums = zeros(size(sums));
for n=1:size(th1,2)
    sums = sums + exp(tr2*th1(:,n));
end
    
for m=1:size(y_tr,2)
    for n=1:size(y_tr,1)
        y_calcr_nm = exp(tr2(n,:)*th1(:,m));
        y_calcr(n,m) = y_calcr_nm/sums(n,1);
    end
end

sums = 0;
for n=1:size(th1,2)
    sums = sums + exp(ts2*th1(:,n));
end
    
for m=1:size(y_ts,2)
    for n=1:size(y_ts,1)
        y_calc_nm = exp(ts2(n,:)*th1(:,m));
        y_calc(n,m) = y_calc_nm/sums(n,1);
    end
end

%set thresholds
indices = find(abs(y_calcr)<0.9);
y_calcr(indices) = 0;
indices = find(abs(y_calc)<0.9);
y_calc(indices) = 0;
indices = find(abs(y_calcr)>0.9);
y_calcr(indices) = 1;
indices = find(abs(y_calc)>0.9);
y_calc(indices) = 1;

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

%create the plot for cross entropy
plot(it,ce,it,ces)
xlabel('Iterations')
ylabel('Cross Entropy')
legend({'Training data','Testing Data'},'Location','southeast')