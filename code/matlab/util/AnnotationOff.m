function AnnotationOff(hh,idx)
if ~exist('idx','var')
    idx = 1:length(hh);
end
for i=1:length(idx)
    x=get(hh(idx(i)),'Annotation');
    x.LegendInformation.IconDisplayStyle='off';
end
