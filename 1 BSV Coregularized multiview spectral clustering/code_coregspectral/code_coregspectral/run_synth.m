clear;


addpath('C:\Users\Student1\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision');
folder = 'C:\Users\Student1\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision\tempdata\';  %location of mydatafile.txt. Could even be remote
 
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

for ppp=1:size(FileNames,1)
    %     ppp=1
    clearvars -except ppp FileNames PathName FileName M_path folder
    M_path = strcat(folder,FileNames{ppp});
    load(M_path);



    fid=fopen(strcat('BSV_',FileNames{ppp},'.csv'),'a');




num_views = length(X);

% numClust = 2;
truth = Y;
numClust = length(unique(Y));



iter = 4;
    
for jj=1:num_views
    sigma(jj) = optSigma(X{jj});
end

lambda = 0.5; 
numiter = 50;

%% single best view
% for iii=1:num_views
% to covert to loop

for p=1:numiter
    for iii=1:num_views
        [V E F P R nmi(iii,p) avgent AR indexes(:,iii)] = baseline_spectral(X{iii},numClust,sigma(iii),truth);
        myresults =ClusteringMeasure1(Y, indexes(:,iii));
        accuracy(iii,p) = myresults(1);
    end
end
temp = sum(accuracy,2);
[val,ind] = max(temp);

for loop = 1:numiter
    [V E F P R nmi_one(loop) avgent AR indexes] = baseline_spectral(X{ind},numClust,sigma(ind),truth);
    myresults(loop,:) =ClusteringMeasure1(Y, indexes);
end

myresults
a_ress(1,:)= mean(myresults,1);
a_ress(2,:)= std(myresults,1);
fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(1,:)');
fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(2,:)');




end % end of for ppp





