------------------------------------------------
clear;
addpath(genpath(cd))




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

addpath('C:\Users\Student1\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision');
for ppp=1:size(FileNames,1)
    %     ppp=1
    clearvars -except ppp FileNames PathName FileName M_path
    folder = 'C:\Users\Student1\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision\tempdata\';  %location of mydatafile.txt. Could even be remote
    M_path = strcat(folder,FileNames{ppp});
    load(M_path);

    fid=fopen(strcat(FileNames{ppp},'.csv'),'a');


    %%

    k = max(Y);
    truth = Y;
    num_views = size(X,2);
    % num_views = 3;
    num_iter = 20;

    fprintf('\nPairwise multiview LRSSC\n');



    if ppp == 1
        opts.mu = 10^2;
        lambda1 = 0.3;
        lambda3 = 0.3;

    elseif ppp == 9
        opts.mu = 10^3;
        lambda1 = 0.7;
        lambda3 = 0.7;
    elseif ppp == 10
        opts.mu = 10^2;
        lambda1 = 0.5;
        lambda3 = 0.7;
    else
        opts.mu = 10^2;
        lambda1 = 0.5;
        lambda3 = 0.5;
    end
    opts.lambda = [lambda1 (1-lambda1) lambda3];
    opts.noisy = true;


    opts.mu = 10^2;
    lambda1 = 0.3;
    lambda3 = 0.3;
    opts.lambda = [lambda1 (1-lambda1) lambda3];
    opts.noisy = true;
    for ii=1:10 % 
        A = pairwise_MLRSSC(X, opts); %
        [best.CA best.F best.P best.R best.nmi best.AR, idx] = spectral_clustering(A, k, truth);

        myresults(ii,:) =ClusteringMeasure1(Y, idx); % 

    end

    a_ress(1,:)= mean(myresults,1);
    a_ress(2,:)= std(myresults,1);
    fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(1,:)');
    fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(2,:)');

end


