function Mat=matScale(mat,rscale,cscale)
% matScale - scale up matrix by size of rcscale
% function Mat=matScale(mat,rscale,cscale) 
% 
% [Input]
%   mat: original matrix
%   rscale: scale up factor for row
%   cscale: scale up factor for column
% 
% [Output]
%   Mat: scaled matrix
% 
% [Usage]
% matScale(magic(3),1:3,1:3)
% 
%%
if ~exist('rscale','var')||isempty(rscale)
    rscale=1;
end
if ~exist('cscale','var')||isempty(cscale)
    cscale=1;
end

[r,c,d] = size(mat);

if length(rscale)==1
    rscale=repmat(rscale,r,1);
end

if length(cscale)==1
    cscale=repmat(cscale,c,1);
end

if length(rscale)~=r || length(cscale)~=c
    error('mat size and r/c scale is not compatible')
end

% increase row
% Mat0=zeros(sum(rscale),c);
% for i = 1:c
%     Mat0(:,i)=vecInc(mat(:,i),rscale);
% end
% Mat=zeros(sum(rscale),sum(cscale));
% for i = 1:sum(rscale)
%     Mat(i,:)=vecInc(Mat0(i,:),cscale);
% end

% increase row
if any(rscale~=1)
    Mat0=cell(length(rscale),1);
    for i = 1:length(rscale)
        idx=(1:rscale(i));
        Mat0{i}(idx,:)=mat(repmat(i,rscale(i),1),:);
    end
    mat=merge(Mat0,1);
end
if any(cscale~=1)
    Mat0=cell(1,length(cscale));
    for i = 1:length(cscale)
        idx=(1:cscale(i));
        Mat0{i}(:,idx)=mat(:,repmat(i,rscale(i),1));
    end
    mat=merge(Mat0,2);
end
Mat=mat;






