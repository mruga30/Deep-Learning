%read all the images, parse them and add them to an array of 154*1600
C = []; y = [];

myFolder = '../yalefaces/yalefaces';
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

%input from user
hidden = input('Enter the nodes of the hidden layers:');

%initialize hyperparameters (lr = learning rate, hdn = numeber of hidden layers, bt=beta, th=theta and wt = hidden layer weights)
lr = 0.2;
hdn = size(hidden,2);
wt{1} = rand(size(tr2,2),hidden(1,1))/30;
for n=1:hdn-1
    wt{n+1} = rand(hidden(1,n),hidden(1,n+1))/30;
end
wt{hdn+1} = rand(hidden(1,hdn),size(y_tr,2))/30;

%initial calculations
gnet{1} = 1./(1+exp(-tr2*wt{1}));
for n=2:hdn+1
    gnet{n} = 1./(1+exp(-gnet{n-1}*wt{n}));
end

delta{hdn+1} = y_tr-gnet{hdn+1};
for n=hdn:-1:1
    delta{n}=(delta{n+1}*wt{n+1}').*(gnet{n}.*(1-gnet{n}));
end

for n=hdn:-1:1
    grad{n+1}= (gnet{n}'*delta{n+1})/size(tr2,1);
end
grad{1}= (tr2'*delta{1})/size(tr2,1);

indices = find(gnet{hdn+1}==0);
gnet{hdn+1}(indices) = 0.0001;
indices = find(gnet{hdn+1}==1);
gnet{hdn+1}(indices) = 0.9999; 

%log likelihood calculations
ll_old = sum(mean(y_tr.*log(gnet{hdn+1})+(1-y_tr).*log(1-gnet{hdn+1})));
ll_new = ll_old; llm =[];
iter = 0; itm =[];

%training the system
while iter < 5000
    for n = 1:hdn+1
        wt{n} = wt{n}+(lr*grad{n});
    end
    
    %if log likehood has decreased, change the learning rate
    if ll_old > ll_new
        lr = 0.94*lr;
    end
    
    %update the parameters according to new theta/beta
    gnet{1} = 1./(1+exp(-tr2*wt{1}));
    for n=2:hdn+1
        gnet{n} = 1./(1+exp(-gnet{n-1}*wt{n}));
    end
    
    delta{hdn+1} = y_tr-gnet{hdn+1};
    for n=hdn:-1:1
        delta{n}=(delta{n+1}*wt{n+1}').*(gnet{n}.*(1-gnet{n}));
    end
    
    for n=hdn:-1:1
        grad{n+1}= (gnet{n}'*delta{n+1})/size(tr2,1);
    end
    grad{1}= (tr2'*delta{1})/size(tr2,1);
    
    %to avoid NaN/Inf
    indices = find(gnet{hdn+1}==0);
    gnet{hdn+1}(indices) = 0.0001;
    indices = find(gnet{hdn+1}==1);
    gnet{hdn+1}(indices) = 0.9999; 
    
    %store values in an array to later plot them
    itm = [itm,iter];
    llm = [llm,ll_old];
    
    %calculate new log likelihood
    ll_old = ll_new;
    ll_new = sum(mean(y_tr.*log(gnet{hdn+1})+(1-y_tr).*log(1-gnet{hdn+1})));
    iter = iter+1;
end
 
%testing the system
gnet_ts{1} = 1./(1+exp(-ts2*wt{1}));
for n=2:hdn+1
    gnet_ts{n} = 1./(1+exp(-gnet_ts{n-1}*wt{n}));
end

gnet_tr{1} = 1./(1+exp(-tr2*wt{1}));
for n=2:hdn+1
    gnet_tr{n} = 1./(1+exp(-gnet_tr{n-1}*wt{n}));
end

%setting thresholds
indices = find(abs(gnet_tr{hdn+1})<0.9);
gnet_tr{hdn+1}(indices) = 0;
indices = find(abs(gnet_tr{hdn+1})>0.9);
gnet_tr{hdn+1}(indices) = 1;
indices = find(abs(gnet_ts{hdn+1})<0.9);
gnet_ts{hdn+1}(indices) = 0;
indices = find(abs(gnet_ts{hdn+1})>0.9);
gnet_ts{hdn+1}(indices) = 1;

%accuracy
vr = 0; vs=0;
for n = 1:size(y_ts,1)
    if isequal(zeros(1,size(gnet_ts{hdn+1},2)),gnet_ts{hdn+1}(n,:))
        gnet_ts{hdn+1}(n,1)=1;
    end
    if isequal(y_ts(n,:),gnet_ts{hdn+1}(n,:))
        vs=vs+1;
    end
end

for n = 1:size(y_tr,1)
    if isequal(y_tr(n,:),gnet_tr{hdn+1}(n,:))
        vr=vr+1;
    end
end

acc_ts = (vs)*100/size(y_ts,1)
acc_tr = (vr)*100/size(y_tr,1)

%plots
plot(itm,llm);