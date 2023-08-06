

clear 
addpath('C:\Users\Student1\OneDrive - Higher Education Commission\Thesis\Actual Phd work\1st paper review\Confirm comparision');
    
% FileNames ={'3sources.mat'; 'BBC.mat'; 'bbcsport_2view.mat';
%     'Caltech101-7.mat';
%     'handwritten.mat';
%     'MSRA_6view.mat';
%     'MSRC.mat';
%     'NGs.mat';
%     'prokaryotic.mat';
%     'UCI_3view.mat';
%     'yale_mtv.mat'
%     };

FileNames ={
   '3sources.mat';
    };


for ppp=1:length(FileNames)
%     ppp=1;
    clearvars -except ppp FileNames PathName FileName M_path folder
    M_path = FileNames{ppp};

    load('3sources.mat');

    




gt = Y;
for i=1:length(X)
    temp = X{i};  %Xv(dv*n)
    X{i} = temp';
end
 
maxIters = 10;
alpha = 0.8;
beta = 0.5;
d = 20; %lower dimension
gamma = 0.004;
tic;
for ii=1:100
    disp('executing ')
    ii
    [result, l] = MCLES(X, alpha, beta, d, gamma, maxIters, gt);
    myresults(ii,:) =ClusteringMeasure1(gt, l)
end
tt=toc;

fid=fopen(strcat(FileNames{ppp},'.csv'),'a');
disp(myresults);
a_ress(1,:)= mean(myresults,1);
a_ress(2,:)= std(myresults,1);
fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(1,:)');
fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(2,:)');
end % end of for ppp
