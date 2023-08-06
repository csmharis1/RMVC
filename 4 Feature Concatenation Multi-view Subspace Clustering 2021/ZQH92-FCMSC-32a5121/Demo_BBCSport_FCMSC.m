clear



FileNames ={'handwritten.mat';
'MSRA_6view.mat';
'MSRC.mat';
'NGs.mat';
'prokaryotic.mat';
'UCI_3view.mat';
'yale_mtv.mat'
};

addpath('C:\Users\Student1\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision');

for ppp=1:size(FileNames,1)
% ppp=1;
clearvars -except ppp FileNames PathName FileName M_path 
folder = 'C:\Users\Student1\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision\tempdata\';  %location of mydatafile.txt. Could even be remote
M_path = strcat(folder,FileNames{ppp});
load(M_path);
fid=fopen(strcat(FileNames{ppp},'.csv'),'a');
X = cellfun(@transpose,X,'UniformOutput',false); 
gt = Y;







addpath('./Tools');

num_views = size(X,2);
num_C = size(unique(gt),1);

opts.lambda_1 = 100;
opts.lambda_2 = 100;

opts.lambda_3 = 0; % zero for FCMSC and non-zero for grFCMSC!
 opts.lambda_3 = 0.001; % 0.001 for lambda_3 used in experiments on BBCSport dataset

opts.rho = 1.9;

for i = 1:2
    fprintf('%d-th iteration\n', i);
    [C,~,~] = FCMSC(X,opts);
    [NMI_c(i),ACC_c(i),F_c(i),RI_c(i), C]=clustering(abs(C)+abs(C'), num_C, gt); % return C is added by me in function too
    myresults(i,:) =ClusteringMeasure1(gt, C); %% added by me
    fprintf('\t--- Clusterint results of C: NMI = %f, ACC = %f, F = %f, RI = %f\n',NMI_c(i),ACC_c(i),F_c(i),RI_c(i));
end




a_ress(1,:)= mean(myresults,1);

fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(1,:)');
a_ress(2,:)= std(myresults,1);
disp('done')
        fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(2,:)');
end