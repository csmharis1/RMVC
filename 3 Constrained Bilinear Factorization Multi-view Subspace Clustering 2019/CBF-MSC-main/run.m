
clear 
FileNames ={'3sources.mat'; 'BBC.mat'; 'bbcsport_2view.mat';
'Caltech101-7.mat';
'handwritten.mat';
'MSRA_6view.mat';
'MSRC.mat';
'NGs.mat';
'prokaryotic.mat';
'UCI_3view.mat';
'yale_mtv.mat'
};




addpath('C:\Users\Chris\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision');
for ppp=1:size(FileNames,1)
clearvars -except ppp FileNames PathName FileName M_path

folder = 'C:\Users\Chris\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision\tempdata\';  %location of mydatafile.txt. Could even be remote
M_path = strcat(folder,FileNames{ppp});
load(M_path);

fid=fopen(strcat(FileNames{ppp},'.csv'),'a');

X = cellfun(@transpose,X,'UniformOutput',false); 
gt = Y;





addpath('./ClusteringMeasure');



%%



numC = size(unique(gt),1);

for i = 1:size(X,2)
    X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1); 
end

NMI_all = [];
ACC_all = [];
F_all = [];
AVG_all = [];
P_all = [];
RI_all = [];
myresults = [];
opts.lambda_1 = 100; 
opts.rho = 1.9;
opts.dim_V = 35; % its the k

fprintf('lambda_1 = %f, dim_V = %d\n', opts.lambda_1,opts.dim_V);
for i = 1:5
    fprintf('Clustering in %d-th iteration\n',i);
    [U,V]  = CBFMSC(X,opts);
    [NMI,ACC,F,AVG,P,RI,C] = ClusteringResults(U,V,gt,numC);
    fprintf('\tNMI: %f, ACC: %f, F: %f, AVG: %f, P: %f, RI: %f\n',NMI,ACC,F,AVG,P,RI);
    NMI_all = [NMI_all, NMI];
    ACC_all = [ACC_all, ACC];
    F_all = [F_all, F];
    AVG_all = [AVG_all, AVG];
    P_all = [P_all, P];
    RI_all = [RI_all, RI];
    myresults(i,:) =ClusteringMeasure1(gt, C);
    
end

myresults

a_ress(1,:)= mean(myresults,1);

    fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(1,:)');
a_ress(2,:)= std(myresults,1);
disp('done')
        fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(2,:)');
end