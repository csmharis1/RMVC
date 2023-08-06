
clear 
currentFolder = pwd;
addpath(genpath(currentFolder));
%% code added by haris
addpath('C:\Users\Chris\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision');
folder = 'C:\Users\Chris\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision\tempdata\';  %location of mydatafile.txt. Could even be remote
    
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

numdata = length(FileNames);




runtimes = 2; 

for ppp=1:size(FileNames,1)

    clearvars -except ppp FileNames PathName FileName M_path folder runtimes
    M_path = strcat(folder,FileNames{ppp});
    load(M_path);

    fid=fopen(strcat(FileNames{ppp},'.csv'),'a');
    X = cellfun(@transpose,X,'UniformOutput',false); % added by me if need to ranspose all the data view in X
 
y0=Y;

c = length(unique(y0));
%% iteration ...
for rtimes = 1:runtimes
    [y, U, S0, S0_initial, F, evs] = GMC(X, c); % c: the # of clusters
    metric = CalcMeasures(y0, y);
    myresults(rtimes,:) =ClusteringMeasure1(y0, y);
    

end

myresults
a_ress(1,:)= mean(myresults,1);
a_ress(2,:)= std(myresults,1);
fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(1,:)');
fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(2,:)');
end
