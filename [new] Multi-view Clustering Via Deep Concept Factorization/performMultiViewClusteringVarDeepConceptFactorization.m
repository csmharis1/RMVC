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
addpath('C:\Users\Student1\OneDrive - Higher Education Commission\Thesis\Actual Phd work\1st paper review\Confirm comparision');


for ppp=1:size(FileNames,1)

clearvars -except ppp FileNames PathName FileName M_path

folder = 'C:\Users\Student1\OneDrive - Higher Education Commission\Thesis\Actual Phd work\1st paper review\Confirm comparision\tempdata\';  %location of mydatafile.txt. Could even be remote
M_path = strcat(folder,FileNames{ppp});
load(M_path);
fid=fopen(strcat(FileNames{ppp},'.csv'),'a');

X = cellfun(@transpose,X,'UniformOutput',false); % added by me
gt = Y;








%% added by me up
addpath(genpath('Dataset/'));
addpath(genpath('Results/'));
dataDirectory = 'Dataset/';
resultDirectory = 'Results/';

if(~exist('Results', 'file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end

dataName = {'3Sources'};
dataName = FileNames; 
numberOfDatasets = length(dataName);

 
    
    options = [];
    options.WeightMode = 'HeatKernel';
    options.tt = 10;
    options.NormWeight = 'NCW';
    options.k = 5;
    options.KernelType = 'Gaussian';
    options.maxIter = 200;
    options.minIter = 20;
    options.round = 1;
    options.repeat = 5;
    options.error = 1e-2;
    options.clusteringFlag = 1;
    options.beta = 100;
    options.gamma = 0.5;
    options.pi = zeros();
    options.PiFlag = 1;
    options.alpha = [1 10];
    layers = [100 50];
    
    odata = ppp;
%     formatData = [dataDirectory, cell2mat(dataName(odata))];
%     load(formatData);
%     end
    fprintf('Performing on dataset: %s\n', cell2mat(dataName(ppp)));
    
    %normalize dataset
    numberOfView = numel(X);
    
    for i = 1 : numberOfView
        
        X{i} = NormalizeData(X{i}, 2);
        options.pi(i) = 1 / numberOfView; 
   end
    
    numberOfCluster = length(unique(Y));
    numberOfFeature = size(X{1}, 2);
    fprintf('Dataset feature: %d\n', numberOfFeature);
    
    ACC = zeros();
    NMI = zeros();
    FScore = zeros();
    ARI = zeros();
    
    Vcon = cell(1, options.repeat);
    %resultLabel = cell(1, options.repeat);
    objectiveFunctionValue = cell(1, options.repeat);
    
    for iter = 1 : options.repeat
        
        fprintf('Perform %d-th times of %d...', iter, options.repeat);
        
        [Vcon{iter}, objectiveFunctionValue{iter}] = ...
        multiViewClusteringVarDeepConceptFactorization(X, options, layers);
        [ACC(iter), NMI(iter), FScore(iter), ARI(iter), indic] = ...
            printResult(Vcon{iter}, Y, numberOfCluster, options.clusteringFlag);
        myresults(iter,:) =ClusteringMeasure1(Y, indic); %% added by me
        
    end
    
    Result = zeros(10, options.repeat);
    
    Result(1,:) = ACC;
    Result(2,:) = NMI;
    Result(3,:) = FScore;
    Result(4,:) = ARI;
    Result(5,1) = mean(ACC);
    Result(5,2) = mean(NMI);
    Result(5,3) = mean(FScore);
    Result(5,4) = mean(ARI);
    Result(6,1) = std(ACC);
    Result(6,2) = std(NMI);
    Result(6,3) = std(FScore);
    Result(6,4) = std(ARI);
    
 fprintf('mean(NMI):%0.4f\n',mean(NMI));
    fprintf('mean(FScore):%0.4f\n',mean(FScore));
    fprintf('mean(ARI):%0.4f\n',mean(ARI));
    




a_ress(1,:)= mean(myresults,1);

fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(1,:)');
a_ress(2,:)= std(myresults,1);
disp('done')
        fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(2,:)');
end









































