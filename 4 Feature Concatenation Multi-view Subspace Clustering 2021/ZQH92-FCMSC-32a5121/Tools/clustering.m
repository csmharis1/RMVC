function [nmi,ACC,f,RI, C]=clustering(S, cls_num, gt)

[C] = SpectralClustering(S,cls_num);
[~, nmi, ~] = compute_nmi(gt,C);
ACC = Accuracy(C,double(gt));
[f,~,~] = compute_f(gt,C);
RI =0;
% [~,RI,~,~]=RandIndex(gt,C);
end
