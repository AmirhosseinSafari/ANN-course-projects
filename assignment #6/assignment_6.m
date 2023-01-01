clc, clearvars, close all

x = -6:0.1:6;
t = sin(x) + x.*cos(3.*x);

t=t+0.1*randn(size(x));

figure(1)
plot(x,t,'o');

[n,m]=size(t);

nn=60;
net = fitnet(nn);
net.trainParam.min_grad=1e-20;
net.trainParam.epochs=1000;
net.trainParam.max_fail=20;
net.Divideparam.trainRatio=0.7;
net.Divideparam.valRatio=0.15;
net.Divideparam.testRatio=0.15;
net = train(net,x,t);
view(net)
y = net(x);
perf = perform(net,y,t);

rng=-6:0.01:6;

IW = net.IW{1};
b1 = net.b{1};
r=rescale(rng',-1,1)';
 
h1 = tansig(IW*r + repmat(b1,1,size(r,2)));
 
LW = net.LW{2};
% LW = rescale(LW',-1,1)';
% b2 = net.b{2};
% h2 = purelin(LW*h1 + repmat(b2,1,size(r,2)));

y2=net(rng);

figure(2)
for i=1:nn
subplot(ceil(nn/3),3,i);plot(rng,h1(i,:));
title(['*',num2str(LW(1,i))])
end

figure(3)
plot(x,t,'o');hold on
plot(rng,y2,'r');hold off