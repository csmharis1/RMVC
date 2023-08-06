
clear;
%% 
% FileNames ={'3sources.mat'; 'BBC.mat'; 'bbcsport_2view.mat';
% 'Caltech101-7.mat';
% 'handwritten.mat';
% 'MSRA_6view.mat';
% 'MSRC.mat';
% 'NGs.mat';
% 'prokaryotic.mat';
% 'UCI_3view.mat';
% 'yale_mtv.mat'
% };

addpath('C:\Users\Student1\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision');
for ppp=1:size(FileNames,1)
%     ppp=1
clearvars -except ppp FileNames PathName FileName M_path
folder = 'C:\Users\Student1\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision\tempdata\';  %location of mydatafile.txt. Could even be remote
M_path = strcat(folder,FileNames{ppp});
load(M_path);




addpath("./evaluate_tools/");


kf_vec = [4,4,3,4,4,4,4,8];
best_init_X = 8;

%% Caltech dataset

multi_X = X; 

for j=1:length(multi_X)
    multi_X{1,j} = normalize_data(multi_X{1,j});
end

 fid=fopen(strcat(FileNames{ppp},'.csv'),'a'); 


maxiter = 2;
run_times = 2; 
kx = max(Y);
n_views = length(multi_X);
w_vec = (zeros(1,n_views)+1)*1/n_views;
lamda = 2.^(-6:5:6);

parameters = lamda;
n_parameters = length(parameters);
records = zeros(n_parameters, 3*2);
a_ress = zeros(2,7);
%% loop
for p=1:n_parameters
    for j=1:run_times
        [indicators, C_haris] = mv_itcc(multi_X,Y,kx,...
            kf_vec,maxiter,w_vec,lamda(p),best_init_X);
        myresults(j,:) =ClusteringMeasure1(Y, C_haris); % added by haris
     
        fprintf('****************parameters**************: %d-----%d\n',p,n_parameters);
        fprintf('*************************run_times**************************: %d\n',j);
    end
    previous = a_ress(1,1);
    a_ress(1,:)= mean(myresults,1);
    a_ress(2,:)= std(myresults,1);
    if previous < a_ress(1,1)
        fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(1,:)');
        fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(2,:)');
    else
        disp("not greater");
    end
    
    
    

end
end

