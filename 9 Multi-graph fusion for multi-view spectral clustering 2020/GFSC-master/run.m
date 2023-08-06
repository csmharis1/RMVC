
clear 
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
%         ppp=1;
    clearvars -except ppp FileNames PathName FileName M_path folder
    M_path = strcat(folder,FileNames{ppp});
    load(M_path);
    fid=fopen(strcat(FileNames{ppp},'.csv'),'a');
    


data = X';
label = Y;
para1=[.01  1];
para2=[100,1000,2000];
para3=[.01];


for i=1:size(data,1)
    dist = max(max(data{i})) - min(min(data{i}));
    m01 = (data{i} - min(min(data{i})))/dist;
    data{i} = 2 * m01 - 1;
end
count =1;
for i=1:length(para1)
    for j=1:length(para2)
        for k=1:length(para3)
            [result, indexes]=multigraph(data,label,para1(i),para2(j),para3(k));
            myresults(count,:) =ClusteringMeasure1(label, indexes);
            count = count+1;
%           dlmwrite('reuters.txt',[para1(i) para2(j) para3(k) result(1,:) result(2,:) result(3,:)   ],'-append','delimiter','\t','newline','pc');
        end
    end
end

myresults
a_ress(1,:)= mean(myresults,1);
a_ress(2,:)= std(myresults,1);
fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(1,:)');
fprintf(fid,'%g,  %g, %g, %g, %g,  %g, %g, \n ',a_ress(2,:)');

end
