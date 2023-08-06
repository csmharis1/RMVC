

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
%         ppp=2
    clearvars -except ppp FileNames PathName FileName M_path
    folder = 'C:\Users\Chris\OneDrive - Higher Education Commission\Thesis\Actual Phd work\my codes all\Confirm comparision\tempdata\';  %location of mydatafile.txt. Could even be remote
    M_path = strcat(folder,FileNames{ppp});
    load(M_path);
    fid=fopen(strcat(FileNames{ppp},'.csv'),'a');
    K = max(Y);
    gnd = Y;
    X = cellfun(@transpose,X,'UniformOutput',false); 
    
    

    
    data = X;

%% deve code
addpath('tools/');
addpath('print/');
options = [];
options.maxIter = 100;
options.error = 1e-6;
options.nRepeat = 10;
options.minIter = 30;
options.meanFitRatio = 0.1;
options.rounds = 30;
options.K=max(Y);
options.Gaplpha=100;                            %Graph regularisation parameter
options.WeightMode='Binary';
options.alpha = 100;

% options.alphas = [0.01 0.01];
options.alphas = zeros(1, length(data))+ 0.01;
options.kmeans = 1;
options.beta=0;




for i = 1:length(data)  %Number of views
%     dtemp=computeDistMat(data{i},2);
%     W{i}=constructW(dtemp,20);
%     data{i} = data{i} / sum(sum(data{i}));
    options.WeightMode='Binary';
    W{i}=constructW_cai(data{i}',options);                      %Incorrect call to construct weight matrix
    %Weight matrix constructed for each view
    data{i} = data{i} / sum(sum(data{i}));
end

U_final = cell(1,3);
V_final = cell(1,3);
V_centroid = cell(1,3);
for i = 1:3
   [U_final{i}, V_final{i}, V_centroid{i}, log, indic] = GMultiNMF(data, K, W,gnd, options);
  [a, indic] =  printResult( V_centroid{i}, gnd, K, options.kmeans);
   myresults(i,:) =ClusteringMeasure1(gnd, indic);
   fprintf('\n');
end



myresults
a_ress(1,:)= mean(myresults,1);
a_ress(2,:)= std(myresults,1);
fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(1,:)');
fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(2,:)');
end % end of for ppp

