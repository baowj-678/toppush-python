clc 
clear 

load spambase.mat

% training the ranking model
opt.lambda = 1;
w = topPush(Xtr, ytr, opt);

% test
pdt = Xte*w; 
