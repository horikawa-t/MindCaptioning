function [d,hw] = cohend(val1,val2,rowcol,p,side)
% function d = cohend(val1,val2)
% calculate cohen's d between two groups
%
%
% [inputs]
%   -val1:values of 1st class
%   -val2:values of 2nd class
%   -rowcol:direction of calculation default=1(row)
%   -p:confidence level (default = 0.95)
%   -side:onesided or twosided (default = onesided)
%
%
% [output]
%   -d: Cohen's d
%   -hw: half width of confidence interval of Cohen's d
%
%
%
% Written by Tomoyasu Horikawa horikawa.t@gmail.com 20231225
%
%
%
if ~exist('rowcol','var') || isempty(rowcol)
    rowcol = 1;
end

if ~exist('p','var') || isempty(p)
    p = 0.95;
end
if ~exist('side','var') || isempty(side)
    side = 'onesided';
end

n1 = size(val1(~isnan(val1)),rowcol);
n2 = size(val2(~isnan(val2)),rowcol);
s = sqrt((nanvar(val1,[],rowcol)*n1+nanvar(val2,[],rowcol)*n2)/(n1+n2));
d = (nanmean(val1,rowcol)-nanmean(val2,rowcol))./s;
sed = sqrt( (n1+n2)/(n1*n2) + (d.^2)./(2*(n1+n2-2)) );

switch side
    case 'onesided'
        c = tinv(p,n1-1);
    case 'twosided'
        c = tinv((1+p)/2,n-1);
    otherwise
        error('Invalid ''side'' option.')
end
hw = c.*sed./sqrt(n1);
