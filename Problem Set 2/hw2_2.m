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

%initialize hyperparameters (lr = learning rate, hdn = hidden layer nodes, bt=beta, th=theta)
lr = 0.2;
hdn = 370; 
bt = rand(size(tr2,2),hdn)/30;
th = rand(hdn,size(y_tr,2))/30;

%initial calculations
gnet_h = 1./(1+exp(-tr2*bt));
gnet_o = 1./(1+exp(-gnet_h*th));
grad_th = gnet_h'*(y_tr-gnet_o);
grad_bt = tr2'*(((y_tr-gnet_o)*th').*(gnet_h.*(1-gnet_h)));

%testing data y_hat initial calculations
y_hats = 1./(1+exp(-(1./(1+exp(-ts2*bt)))*th));
indices = find(y_hats==0);
y_hats(indices) = 0.0001;
indices = find(y_hats==1);
y_hats(indices) = 0.9999; 

%log likelihood calculations
ll_old = sum(mean(y_tr.*log(gnet_o)+(1-y_tr).*log(1-gnet_o)));
ll_olds =  sum(sum(y_ts.*log(y_hats)+(1-y_ts).*log(1-y_hats)));
ll_new = ll_old; llm = []; 
ll_new_ts = ll_olds; lls = [];
iter = 0; itm = [];

%training the system
while iter < 400
    th = th + (lr*grad_th);
    bt = bt + (lr*grad_bt);
    
    %if log likehood has decreased, change the learning rate
    if ll_old > ll_new
        lr = 0.94*lr;
    end
    
    %update the parameters according to new theta/beta
    gnet_h = 1./(1+exp(-tr2*bt));
    gnet_o = 1./(1+exp(-gnet_h*th));
    grad_th = gnet_h'*(y_tr-gnet_o);
    grad_bt = tr2'*(((y_tr-gnet_o)*th').*(gnet_h.*(1-gnet_h)));
        
    %to avoid NaN/Inf
    indices = find(gnet_o==0);
    gnet_o(indices) = 0.0001;
    indices = find(gnet_o==1);
    gnet_o(indices) = 0.9999; 
    
    %testing data y_hat for testing data log likelihood calculations
    y_hats = 1./(1+exp(-(1./(1+exp(-ts2*bt)))*th));
    indices = find(y_hats==0);
    y_hats(indices) = 0.0001;
    indices = find(y_hats==1);
    y_hats(indices) = 0.9999; 
    
    %store values in an array to later plot them
    itm = [itm,iter];
    llm = [llm,ll_old];
    lls = [lls,ll_olds];
    
    %calculate new log likelihood
    ll_old = ll_new;
    ll_new = sum(mean(y_tr.*log(gnet_o)+(1-y_tr).*log(1-gnet_o)));
    ll_olds = ll_new_ts;
    ll_new_ts = sum(sum(y_ts.*log(y_hats)+(1-y_ts).*log(1-y_hats)));
    iter = iter+1;
end
 
%testing the system
gnet_h_ts = 1./(1+exp(-ts2*bt));
gnet_o_ts = 1./(1+exp(-gnet_h_ts*th));
gnet_h_tr = 1./(1+exp(-tr2*bt));
gnet_o_tr = 1./(1+exp(-gnet_h_tr*th)); 

%setting thresholds
indices = find(abs(gnet_o_tr)<0.9);
gnet_o_tr(indices) = 0;
indices = find(abs(gnet_o_tr)>0.9);
gnet_o_tr(indices) = 1;
indices = find(abs(gnet_o_ts)<1);
gnet_o_ts(indices) = 0;
indices = find(abs(gnet_o_ts)>1);
gnet_o_ts(indices) = 1;

%accuracy
vr = 0; vs=0;
for n = 1:size(y_ts,1)
    cl=randi([1,14],1,1);
    if isequal(zeros(1,size(gnet_o_ts,2)),gnet_o_ts(n,:))
        gnet_o_ts(n,cl)=1;
    end
    if isequal(y_ts(n,:),gnet_o_ts(n,:))
        vs=vs+1;
    end
end

for n = 1:size(y_tr,1)
    if isequal(y_tr(n,:),gnet_o_tr(n,:))
        vr=vr+1;
    end
end

acc_ts = (vs)*100/size(y_ts,1)
acc_tr = (vr)*100/size(y_tr,1)

%plots
plot(itm,llm,itm,lls)
xlabel('Iterations')
ylabel('Log Likelihood')
legend({'Training data','Testing Data'},'Location','northeast')

%confusion matrix
cf = zeros(size(y_tr,2),size(y_tr,2));
for n = 1:size(y_ts,1)
    in1 = find(gnet_o_ts(n,:)==1);
    in2 = find(gnet_o_ts(n,:)==1);
    cf(in1,in2) = cf(in1,in2)+1;
end

confusionchart(cf)