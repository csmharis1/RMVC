clear;


%% added by me
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

for ppp=1:size(FileNames,1)
    %     ppp=1
    clearvars -except ppp FileNames PathName FileName M_path folder
    M_path = strcat(folder,FileNames{ppp});
    load(M_path);

    %     X = cellfun(@transpose,X,'UniformOutput',false); % added by me if need to ranspose all the data view in X


    fid=fopen(strcat('M_',FileNames{ppp},'.csv'),'a');

    num_views = length(X);


    truth = Y;
    numClust = length(unique(Y));

    for jj=1:num_views
        sigma(jj) = optSigma(X{jj});
    end


    numiter = 30;



    lambda = zeros(1, num_views) + 0.5;

    [F, P, R, nmi, avgent, AR ,myresults] = spectral_centroid_multiview(X,num_views,numClust,sigma,lambda,truth,numiter);
%     myresults(loop,:) =ClusteringMeasure1(Y, indexes);


    myresults
    a_ress(1,:)= mean(myresults,1);
    a_ress(2,:)= std(myresults,1);
    fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(1,:)');
    fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(2,:)');



end % end of for ppp




%% two views
% feature concat
% fprintf('Running with the feature concatenation of two views\n');
% [E F P R nmi avgent] = baseline_spectral([X1 X2],numClust,optSigma([X1 X2]),truth);
% %de sa mindis
% fprintf('Running with the De Sa MinDis\n');
% [F P R nmi avgent AR] = spectral_mindis(X1, X2, numClust,sigma1,sigma2, truth);
% fprintf('Our approach with 2 views (pairwise)\n');
% [U U1 F P R nmi avgent AR] = spectral_pairwise(X1,X2,numClust,sigma1,sigma2,lambda,truth,numiter);
% fprintf('Our approach with 2 views (centroid)\n');
% lambda1 = 0.5; lambda2 = 0.5;
% [U U1 F P R nmi avgent AR] = spectral_centroid(X1,X2,numClust,sigma1,sigma2,lambda1,lambda2,truth,numiter);


%% three views
% fprintf('Running with the feature concatenation of three views\n');
% [E F P R nmi avgent] = baseline_spectral([X1 X2 X3],numClust,optSigma([X1 X2 X3]),truth);

% multiview spectral (pairwise): more than 2 views
% fprintf('Multiview spectral with 3 views\n');
% [F P R nmi avgent AR] = spectral_pairwise_multview(X,num_views,numClust,sigma,lambda,truth,numiter);

% multiview spectral (centroid): more than 2 views
% fprintf('Multiview spectral with 3 views\n');
% lambda = zeros(1, num_views) + 0.5;
% [F P R nmi avgent AR] = spectral_centroid_multiview(X,num_views,numClust,sigma,lambda,truth,numiter);



