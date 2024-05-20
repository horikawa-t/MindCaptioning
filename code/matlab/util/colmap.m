function map = colmap(col,inv)
% create colormap
%
% make colormaps based on pycortex:
%    https://gallantlab.github.io/colormaps.html
%
% colorset
%  {'Paired','viridis','terrain','rainbow','Spectral','Pastel1_r','RdBu_r','coolwarm'}
%
%  inv: if 1 inverse color map;
%
% how to make
% scp -r ~/Desktop/plasma.png root@129.60.220.222:/export/psi/horikawa-t/figicon/
% a=imresize(imread('/home/psi/horikawa-t/figicon/magma.png'),[1,64]);
% a=imread('/home/psi/horikawa-t/figicon/magma.png');
% a=imresize(imread('/home/psi/horikawa-t/figicon/coolwarm_r.png'),[1,64]);
% a=imread('/home/psi/horikawa-t/figicon/pycortex_colormap/RdBu_r.png');
% for i = 1:3
%     for j = 1:size(a,2)
%         fprintf('%d,',a(1,j,i))
%     end
%     fprintf(';\n')
% end
% 
% Written by horikawa-t 20200225

%% color interpolation

switch col
    case 'RGrB_tsi_r'
        map = [...
            2,7,13,19,25,31,37,43,48,53,58,64,70,75,81,86,92,98,103,109,114,120,126,131,137,142,148,154,159,165,170,176,179,180,181,182,183,184,185,186,187,189,190,191,192,193,194,195,196,197,199,200,201,202,203,206,211,217,223,229,235,241,247,253,;
            2,7,13,19,25,31,37,43,48,53,58,64,70,75,81,86,92,98,103,109,114,120,126,131,137,142,148,154,159,165,170,176,176,170,165,159,154,148,142,137,131,126,120,114,109,103,98,92,86,81,75,70,64,58,53,48,43,37,31,25,19,13,7,2,;
            253,247,241,235,229,223,217,211,206,203,202,201,200,199,197,196,195,194,193,192,191,190,189,187,186,185,184,183,182,181,180,179,176,170,165,159,154,148,142,137,131,126,120,114,109,103,98,92,86,81,75,70,64,58,53,48,43,37,31,25,19,13,7,2,;
            ]'/255;
        map = map(end:-1:1,:);
    case 'RGrB_tsi'
        map = [...
            2,7,13,19,25,31,37,43,48,53,58,64,70,75,81,86,92,98,103,109,114,120,126,131,137,142,148,154,159,165,170,176,179,180,181,182,183,184,185,186,187,189,190,191,192,193,194,195,196,197,199,200,201,202,203,206,211,217,223,229,235,241,247,253,;
            2,7,13,19,25,31,37,43,48,53,58,64,70,75,81,86,92,98,103,109,114,120,126,131,137,142,148,154,159,165,170,176,176,170,165,159,154,148,142,137,131,126,120,114,109,103,98,92,86,81,75,70,64,58,53,48,43,37,31,25,19,13,7,2,;
            253,247,241,235,229,223,217,211,206,203,202,201,200,199,197,196,195,194,193,192,191,190,189,187,186,185,184,183,182,181,180,179,176,170,165,159,154,148,142,137,131,126,120,114,109,103,98,92,86,81,75,70,64,58,53,48,43,37,31,25,19,13,7,2,;
            ]'/255;
    case 'Paired'
        map = [... % Paired
            166,142,119,96,72,49,36,61,86,112,137,169,163,141,119,97,75,53,80,115,149,184,218,250,246,242,238,234,230,228,232,237,242,247,251,253,253,253,254,254,254,246,237,228,219,209,199,183,166,150,133,116,114,146,172,198,224,249,244,230,217,203,190,177;...
            206,191,176,161,146,131,123,141,159,176,194,216,215,204,194,183,172,161,159,158,157,156,154,152,130,108,86,64,42,33,62,90,126,154,183,182,171,160,149,138,127,135,144,152,161,170,175,155,134,114,94,74,72,114,147,181,214,248,232,203,174,146,117,89;...
            227,218,210,202,194,186,178,171,164,156,149,140,127,111,94,78,62,46,60,79,97,116,135,151,129,108,86,65,43,31,46,60,78,92,107,97,77,58,39,20,1,34,71,108,145,182,212,202,191,181,171,160,153,153,153,153,153,153,137,117,98,78,59,40;...
            ]'/255;
    case 'plasma'
        map = [...
            17,28,36,44,52,58,65,72,78,85,91,98,104,110,116,122,128,134,139,145,151,156,161,166,171,176,180,185,189,193,197,201,205,208,212,215,218,222,225,228,231,233,236,238,241,243,245,247,248,250,251,252,253,253,253,253,253,252,251,249,247,245,242,240,;
            7,6,5,4,4,4,3,2,2,1,0,0,0,0,0,2,3,6,10,15,18,23,27,32,37,41,46,50,55,59,64,68,73,77,82,86,91,96,100,105,110,115,119,124,130,135,140,145,151,157,162,168,174,180,186,192,199,205,212,218,225,232,239,246,;
            136,141,145,148,151,154,157,159,162,163,165,166,167,168,168,168,167,166,164,162,160,157,154,151,148,144,141,137,132,129,126,121,118,115,111,107,104,100,97,93,90,87,83,80,76,73,69,66,62,59,56,52,49,46,43,40,38,37,36,36,36,38,38,36,;
            ]'/255;
    case 'magma'
        map = [...
0,0,0,1,1,1,2,2,3,4,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,26,27,28,30,31,32,34,35,37,38,40,42,43,45,47,48,50,52,53,55,57,59,60,62,64,66,67,69,71,72,74,75,77,79,80,82,83,85,87,88,90,91,93,94,96,97,99,101,102,104,105,107,108,110,111,113,115,116,118,119,121,122,124,126,127,129,130,132,133,135,137,138,140,141,143,145,146,148,149,151,153,154,156,158,159,161,163,164,166,167,169,171,172,174,176,177,179,181,182,184,185,187,189,190,192,194,195,197,198,200,202,203,205,206,208,209,211,212,214,215,217,218,220,221,222,224,225,226,228,229,230,231,232,234,235,236,237,238,238,239,240,241,242,243,243,244,245,245,246,246,247,247,248,248,249,249,249,250,250,250,251,251,251,251,252,252,252,252,252,253,253,253,253,253,253,253,253,253,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,253,253,253,253,253,253,253,253,253,253,253,253,252,252,252,252,252,252,252,252,252,252,252,251,251,251,251,;
0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,7,7,8,9,10,10,11,12,12,13,13,14,14,15,15,16,16,16,16,17,17,17,17,17,17,17,17,17,16,16,16,16,16,15,15,15,15,15,15,15,15,15,15,15,16,16,16,17,17,18,18,19,19,20,21,21,22,23,23,24,24,25,26,26,27,28,28,29,30,30,31,31,32,33,33,34,34,35,36,36,37,37,38,38,39,40,40,41,41,42,42,43,43,44,44,45,45,46,46,47,47,48,48,49,49,50,51,51,52,52,53,53,54,54,55,55,56,57,57,58,58,59,60,60,61,62,62,63,64,65,66,66,67,68,69,70,71,72,73,74,75,76,77,78,80,81,82,84,85,86,88,89,91,93,94,96,97,99,101,103,104,106,108,110,112,113,115,117,119,121,123,125,127,128,130,132,134,136,138,140,142,144,146,147,149,151,153,155,157,159,161,162,164,166,168,170,172,174,175,177,179,181,183,185,187,188,190,192,194,196,198,199,201,203,205,207,209,210,212,214,216,218,220,221,223,225,227,229,230,232,234,236,238,240,241,243,245,247,249,250,252,;
3,4,6,7,9,11,13,15,17,19,21,23,25,27,29,31,34,36,38,40,42,44,47,49,51,53,56,58,60,63,65,68,70,73,75,77,80,82,85,87,89,92,94,96,98,101,103,104,106,108,110,111,113,114,115,116,117,118,119,120,121,121,122,123,123,124,124,125,125,126,126,126,126,127,127,127,127,128,128,128,128,128,128,129,129,129,129,129,129,129,129,129,129,129,129,129,129,129,129,129,129,129,128,128,128,128,128,128,128,127,127,127,127,126,126,126,126,125,125,125,124,124,123,123,123,122,122,121,121,120,120,119,119,118,117,117,116,116,115,114,114,113,112,112,111,110,109,109,108,107,106,105,105,104,103,102,102,101,100,99,98,98,97,96,96,95,95,94,93,93,93,92,92,92,91,91,91,91,91,91,91,92,92,92,92,93,93,94,94,95,96,96,97,98,99,99,100,101,102,103,104,105,106,107,108,110,111,112,113,115,116,117,118,120,121,123,124,125,127,128,130,131,133,134,136,137,139,141,142,144,146,147,149,151,152,154,156,157,159,161,163,165,166,168,170,172,174,176,177,179,181,183,185,187,189,191,;
%             0,2,4,7,11,15,20,25,30,36,42,49,56,63,70,76,83,89,95,101,107,114,120,126,133,139,145,152,159,165,171,178,185,191,198,204,210,216,222,227,232,236,240,243,246,248,249,250,252,252,253,253,254,254,254,254,254,253,253,253,252,252,252,251,;
%             0,1,3,6,9,11,14,15,17,17,17,16,15,15,15,17,18,21,24,26,29,31,34,36,38,40,43,45,46,49,51,53,55,57,60,63,66,69,73,78,83,89,95,102,109,116,124,131,139,147,154,162,169,176,184,191,199,206,213,221,228,235,242,250,;
%             5,12,20,28,37,45,54,64,74,83,93,102,109,115,119,122,124,126,127,128,128,129,129,129,129,129,128,127,126,125,124,122,120,118,115,113,110,107,104,100,97,95,93,91,91,92,93,95,98,101,105,110,115,120,126,132,138,145,151,158,165,173,180,188,;
            ]'/255;
    case 'YlGnBu'
        map = [...
            254,251,249,247,245,242,240,238,235,230,225,220,216,211,206,201,195,186,177,168,159,149,140,131,123,115,107,100,92,84,76,68,62,58,53,49,44,40,35,30,29,29,30,31,31,32,33,33,34,34,35,35,35,36,36,36,35,30,27,24,20,16,12,9,;
            254,253,253,251,251,250,249,248,247,245,243,241,239,238,236,234,231,228,224,220,217,213,210,206,203,200,197,195,192,189,186,183,179,175,170,165,161,156,151,147,141,135,128,122,116,109,103,96,91,85,80,75,69,64,59,54,50,47,44,41,38,35,32,30,;
            215,209,205,200,195,190,184,179,177,177,177,178,178,179,179,179,180,181,182,182,184,184,185,186,187,188,189,190,192,193,194,195,195,195,194,194,193,193,192,192,190,187,184,181,178,175,172,169,166,164,161,159,156,154,151,149,143,136,128,121,113,105,98,90,;
            ]'/255;
    case 'plasma_r'
        map = [...
            17,28,36,44,52,58,65,72,78,85,91,98,104,110,116,122,128,134,139,145,151,156,161,166,171,176,180,185,189,193,197,201,205,208,212,215,218,222,225,228,231,233,236,238,241,243,245,247,248,250,251,252,253,253,253,253,253,252,251,249,247,245,242,240,;
            7,6,5,4,4,4,3,2,2,1,0,0,0,0,0,2,3,6,10,15,18,23,27,32,37,41,46,50,55,59,64,68,73,77,82,86,91,96,100,105,110,115,119,124,130,135,140,145,151,157,162,168,174,180,186,192,199,205,212,218,225,232,239,246,;
            136,141,145,148,151,154,157,159,162,163,165,166,167,168,168,168,167,166,164,162,160,157,154,151,148,144,141,137,132,129,126,121,118,115,111,107,104,100,97,93,90,87,83,80,76,73,69,66,62,59,56,52,49,46,43,40,38,37,36,36,36,38,38,36,;
            ]'/255;
        map = map(end:-1:1,:);
    case 'magma_r'
        map = [...
            0,2,4,7,11,15,20,25,30,36,42,49,56,63,70,76,83,89,95,101,107,114,120,126,133,139,145,152,159,165,171,178,185,191,198,204,210,216,222,227,232,236,240,243,246,248,249,250,252,252,253,253,254,254,254,254,254,253,253,253,252,252,252,251,;
            0,1,3,6,9,11,14,15,17,17,17,16,15,15,15,17,18,21,24,26,29,31,34,36,38,40,43,45,46,49,51,53,55,57,60,63,66,69,73,78,83,89,95,102,109,116,124,131,139,147,154,162,169,176,184,191,199,206,213,221,228,235,242,250,;
            5,12,20,28,37,45,54,64,74,83,93,102,109,115,119,122,124,126,127,128,128,129,129,129,129,129,128,127,126,125,124,122,120,118,115,113,110,107,104,100,97,95,93,91,91,92,93,95,98,101,105,110,115,120,126,132,138,145,151,158,165,173,180,188,;
            ]'/255;
        map = map(end:-1:1,:);
    case 'YlGnBu_r'
        map = [...
            254,251,249,247,245,242,240,238,235,230,225,220,216,211,206,201,195,186,177,168,159,149,140,131,123,115,107,100,92,84,76,68,62,58,53,49,44,40,35,30,29,29,30,31,31,32,33,33,34,34,35,35,35,36,36,36,35,30,27,24,20,16,12,9,;
            254,253,253,251,251,250,249,248,247,245,243,241,239,238,236,234,231,228,224,220,217,213,210,206,203,200,197,195,192,189,186,183,179,175,170,165,161,156,151,147,141,135,128,122,116,109,103,96,91,85,80,75,69,64,59,54,50,47,44,41,38,35,32,30,;
            215,209,205,200,195,190,184,179,177,177,177,178,178,179,179,179,180,181,182,182,184,184,185,186,187,188,189,190,192,193,194,195,195,195,194,194,193,193,192,192,190,187,184,181,178,175,172,169,166,164,161,159,156,154,151,149,143,136,128,121,113,105,98,90,;
            ]'/255;
        map = map(end:-1:1,:);
    case 'viridis_r'
        map = [... % viridis
            68,69,70,71,71,72,72,71,71,70,69,67,65,63,61,60,58,56,54,52,50,48,46,45,43,42,40,39,37,36,34,33,31,31,30,30,31,32,35,38,42,47,53,59,66,73,81,89,98,107,116,126,136,149,159,170,181,191,202,212,223,233,243,253;...
            1,6,12,18,24,29,34,39,44,49,54,60,65,69,74,78,83,87,91,95,99,103,107,111,115,119,122,126,130,134,137,141,146,150,153,157,161,165,168,172,176,179,183,186,190,193,196,199,202,205,208,210,213,215,217,219,221,223,224,225,227,228,229,231;...
            84,90,95,101,106,111,115,119,123,126,129,132,134,135,137,138,139,140,140,141,141,141,142,142,142,142,142,142,142,141,141,140,140,139,138,136,135,133,131,129,126,123,120,117,113,109,104,100,95,89,84,78,71,63,56,50,43,36,30,26,24,25,30,36;...
            ]'/255;
    case 'viridis'
        map = [... % viridis
            68,69,70,71,71,72,72,71,71,70,69,67,65,63,61,60,58,56,54,52,50,48,46,45,43,42,40,39,37,36,34,33,31,31,30,30,31,32,35,38,42,47,53,59,66,73,81,89,98,107,116,126,136,149,159,170,181,191,202,212,223,233,243,253;...
            1,6,12,18,24,29,34,39,44,49,54,60,65,69,74,78,83,87,91,95,99,103,107,111,115,119,122,126,130,134,137,141,146,150,153,157,161,165,168,172,176,179,183,186,190,193,196,199,202,205,208,210,213,215,217,219,221,223,224,225,227,228,229,231;...
            84,90,95,101,106,111,115,119,123,126,129,132,134,135,137,138,139,140,140,141,141,141,142,142,142,142,142,142,142,141,141,140,140,139,138,136,135,133,131,129,126,123,120,117,113,109,104,100,95,89,84,78,71,63,56,50,43,36,30,26,24,25,30,36;...
            ]'/255;
    case 'terrain'
        map = [... % terrain
            51,45,40,35,29,24,19,13,8,3,0,0,0,0,0,0,5,21,37,53,69,85,101,117,133,149,165,181,197,213,229,245,250,242,234,226,218,210,202,194,186,178,170,162,154,146,138,130,133,141,149,157,165,175,183,191,199,207,215,223,231,239,247,255;...
            51,61,72,83,93,104,115,125,136,147,156,166,174,182,190,198,205,208,211,214,217,221,224,227,230,233,237,240,243,246,249,253,248,238,228,217,207,197,187,176,166,156,146,135,125,115,105,95,98,109,119,129,139,152,162,173,183,193,203,214,224,234,244,255;...
            153,163,174,185,195,206,217,227,238,249,244,214,190,166,142,118,103,106,109,112,115,119,122,125,128,131,135,138,141,144,147,151,150,145,141,137,133,128,124,120,115,111,107,102,98,94,89,85,91,102,112,123,134,147,158,169,179,190,201,212,222,233,244,255;...
            ]'/255;
    case 'rainbow'
        map = [... % rainbow
            127,119,111,103,95,87,79,71,63,55,47,37,29,21,13,5,2,10,18,26,34,42,50,58,66,74,82,90,98,106,114,122,132,140,148,156,164,172,180,188,196,204,212,220,228,236,244,252,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255;...
            0,12,25,37,49,62,74,86,97,109,120,134,144,154,164,174,183,191,199,207,214,220,226,232,237,241,245,248,250,252,254,254,254,254,252,250,248,245,241,237,232,226,220,214,207,199,191,183,174,164,154,144,134,120,109,97,86,74,62,49,37,25,12,0;...
            255,254,254,254,253,253,252,251,250,248,247,245,243,241,239,237,234,232,229,226,223,220,217,214,210,207,203,199,195,191,187,183,177,172,168,163,158,153,148,143,138,132,127,122,116,110,105,99,93,87,81,75,69,62,56,49,43,37,31,25,18,12,6,0;...
            ]'/255;
    case 'Spectral'
        map = [... % Spectral
            158,166,175,183,192,201,209,216,220,225,230,236,241,244,246,247,248,250,251,253,253,253,253,253,253,253,254,254,254,254,254,254,252,248,244,240,236,232,227,218,209,199,190,181,172,161,150,139,129,118,107,97,89,79,71,63,55,52,59,66,73,80,87,94;...
            1,10,20,29,39,48,58,66,73,81,88,97,105,114,124,134,144,154,165,174,182,190,198,206,214,222,227,232,237,242,247,251,254,252,250,249,247,246,244,240,236,232,229,225,221,217,213,208,204,200,196,189,180,168,159,150,141,132,123,114,105,96,87,79;...
            66,68,70,72,74,76,78,77,75,74,72,69,67,69,74,78,83,88,92,97,104,111,117,124,130,137,145,153,161,169,177,185,187,181,174,168,162,156,152,154,156,158,160,161,163,164,164,164,164,164,164,166,170,175,179,182,186,187,183,178,174,170,166,162;...
            ]'/255;
    case 'Spectral_r'
        map = [... % Spectral
            158,166,175,183,192,201,209,216,220,225,230,236,241,244,246,247,248,250,251,253,253,253,253,253,253,253,254,254,254,254,254,254,252,248,244,240,236,232,227,218,209,199,190,181,172,161,150,139,129,118,107,97,89,79,71,63,55,52,59,66,73,80,87,94;...
            1,10,20,29,39,48,58,66,73,81,88,97,105,114,124,134,144,154,165,174,182,190,198,206,214,222,227,232,237,242,247,251,254,252,250,249,247,246,244,240,236,232,229,225,221,217,213,208,204,200,196,189,180,168,159,150,141,132,123,114,105,96,87,79;...
            66,68,70,72,74,76,78,77,75,74,72,69,67,69,74,78,83,88,92,97,104,111,117,124,130,137,145,153,161,169,177,185,187,181,174,168,162,156,152,154,156,158,160,161,163,164,164,164,164,164,164,166,170,175,179,182,186,187,183,178,174,170,166,162;...
            ]'/255;
        map = map(end:-1:1,:);
    case 'Pastel1_r'
        map = [... % Pastel1_r
            242,243,244,246,247,248,250,251,252,249,246,243,240,237,234,231,230,233,236,239,243,246,249,252,254,254,254,254,254,254,254,254,251,247,243,239,235,231,227,223,220,218,216,213,211,209,206,204,201,198,195,192,189,185,182,179,187,196,205,214,223,232,241,251;...
            242,238,235,232,229,226,223,220,217,217,217,217,216,216,216,216,217,222,227,232,237,242,246,251,253,248,243,239,234,229,224,219,215,214,212,210,208,207,205,203,205,209,213,217,221,225,229,233,232,228,224,221,217,212,208,205,201,198,195,192,189,186,183,180;...
            242,241,240,239,238,238,237,236,235,229,224,216,210,204,198,193,189,191,193,195,197,199,200,202,202,197,192,188,183,178,173,168,170,178,186,194,201,209,217,225,225,221,217,213,209,205,202,198,199,203,207,210,214,219,223,226,220,213,207,200,193,187,180,174;...
            ]'/255;
    case 'RdBu_r'
        map = [... % RdBu_r
5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,35,36,37,39,40,41,43,44,45,47,48,49,51,52,53,55,56,57,59,60,61,63,64,65,67,70,73,76,79,82,85,88,91,94,97,101,104,107,110,113,116,119,122,125,128,132,135,138,141,144,147,149,152,154,157,159,162,164,167,169,171,174,176,179,181,184,186,189,191,194,196,199,201,204,206,209,210,211,213,214,216,217,219,220,222,223,225,226,228,229,231,232,234,235,237,238,240,241,243,244,246,247,247,247,247,248,248,248,248,249,249,249,249,249,250,250,250,250,251,251,251,251,252,252,252,252,253,252,252,251,251,251,250,250,250,249,249,249,248,248,248,247,247,247,246,246,245,245,245,244,244,244,243,242,241,239,238,237,236,235,234,232,231,230,229,228,226,225,224,223,222,221,219,218,217,216,215,214,212,211,209,208,206,205,204,202,201,199,198,197,195,194,192,191,190,188,187,185,184,182,181,180,178,176,173,170,167,164,161,158,155,153,150,147,144,141,138,135,132,129,126,123,120,117,114,111,108,105,103,;
48,50,52,54,56,58,60,62,64,67,69,71,73,75,77,79,81,84,86,88,90,92,94,96,98,100,102,104,106,108,109,111,113,115,117,118,120,122,124,125,127,129,131,132,134,136,138,139,141,143,145,147,148,150,152,154,156,158,160,162,164,166,168,170,172,174,176,178,180,182,184,186,188,190,192,194,196,197,198,200,201,202,203,205,206,207,208,210,211,212,213,215,216,217,218,220,221,222,223,225,226,227,229,229,230,231,231,232,233,233,234,235,236,236,237,238,238,239,240,241,241,242,243,243,244,245,245,246,246,245,244,243,242,240,239,238,237,236,235,234,233,232,231,229,228,227,226,225,224,223,222,221,220,219,216,214,212,210,208,206,204,202,199,197,195,193,191,189,187,185,183,180,178,176,174,172,170,168,166,163,160,158,155,152,150,147,144,142,139,136,133,131,128,125,123,120,117,114,112,109,106,104,101,98,96,93,90,87,84,81,79,76,73,70,67,64,62,59,56,53,50,48,45,42,39,36,33,31,28,25,23,22,21,20,19,18,17,16,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,;
97,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147,149,152,155,158,161,164,167,170,172,173,174,175,176,176,177,178,179,180,181,182,183,184,185,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,213,214,215,216,217,218,219,220,221,222,223,223,224,225,225,226,227,228,228,229,230,230,231,232,232,233,234,235,235,236,237,237,238,239,240,240,240,240,241,241,241,241,242,242,242,243,243,243,243,244,244,244,244,245,245,245,246,246,246,246,246,244,242,240,238,236,234,232,231,229,227,225,223,221,219,217,215,214,212,210,208,206,204,202,200,199,196,193,190,188,185,182,180,177,174,171,169,166,163,161,158,155,153,150,147,144,142,139,136,134,131,128,126,124,122,120,118,116,114,112,110,108,106,104,101,99,97,95,93,91,89,87,85,83,81,79,77,75,74,73,71,70,69,67,66,65,63,62,61,59,58,57,55,54,53,51,50,49,47,46,45,43,42,42,41,41,40,40,39,39,39,38,38,37,37,36,36,35,35,34,34,33,33,32,32,31,31,31,;
%             5,9,13,18,22,26,31,36,41,47,52,59,64,73,85,97,110,122,135,147,157,167,176,186,196,206,213,219,225,231,237,243,247,248,249,250,251,252,252,251,249,248,247,245,244,239,235,230,225,221,216,211,205,198,192,187,181,173,161,150,138,126,114,103;...
%             48,56,64,73,81,90,98,106,113,120,127,136,143,150,158,166,174,182,190,197,202,207,212,217,222,227,231,233,236,239,242,245,244,239,235,231,226,222,216,208,199,191,183,174,166,155,144,133,123,112,101,90,79,64,53,42,31,22,18,15,11,7,3,0;...
%             97,108,120,132,144,155,167,174,177,181,185,189,193,197,201,205,209,214,218,222,225,228,230,233,236,239,240,241,243,244,245,246,242,234,227,219,212,204,196,185,174,163,153,142,131,122,114,106,97,89,81,74,69,62,57,51,46,42,40,38,36,34,32,31;...
            ]'/255;
    case 'RdGy_r'
        map = [... % RdGy_r
            26 34 42 50 58 66 74 82 91 100 109 121 130 139 147 155 163 171 179 186 192 198 204 210 216 222 227 232 237 242 247 251 254 254 254 253 253 253 252 251 249 248 247 245 244 239 235 230 225 221 216 211 205 198 192 187 181 173 161 150 138 126 114 103 ;...
            26 34 42 50 58 66 74 82 91 100 109 121 130 139 147 155 163 171 179 186 192 198 204 210 216 222 227 232 237 242 247 251 251 245 240 234 228 223 216 208 199 191 183 174 166 155 144 133 123 112 101 90 79 64 53 42 31 22 18 15 11 7 3 0 ;...
            26 34 42 50 58 66 74 82 91 100 109 121 130 139 147 155 163 171 179 186 192 198 204 210 216 222 227 232 237 242 247 251 249 240 231 223 214 205 196 185 174 163 153 142 131 122 114 106 97 89 81 74 69 62 57 51 46 42 40 38 36 34 32 31 ;...
            ]'/255;
    case 'coolwarm'
        map = [... % coolwarm
            58,63,67,72,77,82,87,92,97,103,108,115,120,126,131,137,142,148,153,159,164,170,175,180,185,190,195,200,205,209,214,218,223,227,231,234,237,239,242,243,245,246,246,247,247,246,245,244,243,241,238,236,233,229,225,221,217,212,207,202,197,191,185,179;...
            76,83,90,96,103,110,117,123,130,136,142,149,155,161,166,172,177,181,186,190,194,198,202,205,208,211,213,215,217,218,219,220,219,217,214,211,208,205,201,197,193,188,183,178,173,167,161,155,149,142,135,128,121,112,104,96,88,79,70,61,50,40,22,3;...
            192,198,204,209,215,220,225,229,234,237,241,244,247,249,251,252,253,254,254,254,254,253,251,250,248,245,242,239,236,232,228,223,217,211,205,199,193,187,181,175,168,162,156,149,143,137,130,124,118,112,106,100,94,87,82,76,71,66,61,56,51,46,42,38;...
            ]'/255;
%     case 'coolwarm'
%         map = [...
%             60,64,69,74,79,84,89,94,99,105,110,115,121,126,132,137,143,149,154,160,165,170,176,181,186,191,196,201,205,210,214,219,223,227,230,234,237,239,241,243,245,246,246,247,247,246,246,244,243,241,239,236,233,230,227,223,219,214,209,204,199,194,187,181,;
%             78,85,92,99,106,113,119,126,132,138,145,150,156,162,167,172,177,182,187,191,195,199,202,205,208,211,213,215,217,218,219,220,219,217,215,212,209,205,202,198,194,189,184,179,174,168,162,156,150,143,136,129,122,115,107,99,91,83,74,65,55,44,29,10,;
%             194,200,206,212,217,222,227,231,235,239,242,245,247,250,251,252,254,254,254,254,254,252,251,250,247,245,242,239,236,232,227,222,217,212,206,200,194,188,182,175,169,163,156,150,144,138,131,125,119,113,107,101,95,89,84,78,73,68,63,58,53,48,43,39,;
%             ]'/255;
    case 'coolwarm_r'
        map = [...
            181,187,194,199,204,209,214,219,223,227,230,233,236,239,241,243,244,246,246,247,247,246,246,245,243,241,239,237,234,230,227,223,219,214,210,205,201,196,191,186,181,176,170,165,160,154,149,143,137,132,126,121,115,110,105,99,94,89,84,79,74,69,64,60,;
            10,29,44,55,65,74,83,91,99,107,115,122,129,136,143,150,156,162,168,174,179,184,189,194,198,202,205,209,212,215,217,219,220,219,218,217,215,213,211,208,205,202,199,195,191,187,182,177,172,167,162,156,150,145,138,132,126,119,113,106,99,92,85,78,;
            39,43,48,53,58,63,68,73,78,84,89,95,101,107,113,119,125,131,138,144,150,156,163,169,175,182,188,194,200,206,212,217,222,227,232,236,239,242,245,247,250,251,252,254,254,254,254,254,252,251,250,247,245,242,239,235,231,227,222,217,212,206,200,194,;
            ]'/255;
    otherwise
        fprintf('%s is not matched.\nUse Paired instead.\n',col)
        map = [... % Paired
            166,142,119,96,72,49,36,61,86,112,137,169,163,141,119,97,75,53,80,115,149,184,218,250,246,242,238,234,230,228,232,237,242,247,251,253,253,253,254,254,254,246,237,228,219,209,199,183,166,150,133,116,114,146,172,198,224,249,244,230,217,203,190,177;...
            206,191,176,161,146,131,123,141,159,176,194,216,215,204,194,183,172,161,159,158,157,156,154,152,130,108,86,64,42,33,62,90,126,154,183,182,171,160,149,138,127,135,144,152,161,170,175,155,134,114,94,74,72,114,147,181,214,248,232,203,174,146,117,89;...
            227,218,210,202,194,186,178,171,164,156,149,140,127,111,94,78,62,46,60,79,97,116,135,151,129,108,86,65,43,31,46,60,78,92,107,97,77,58,39,20,1,34,71,108,145,182,212,202,191,181,171,160,153,153,153,153,153,153,137,117,98,78,59,40;...
            ]'/255;
end

if exist('inv','var') && inv
    map = map(end:-1:1,:);
end

%% debug
if 0
    clc
    idx = round(linspace(1,256,64));
    
    close all
    figure;
    cnt = 0;
    colNames = {'Paired','viridis','terrain','rainbow','Spectral',...
        'Pastel1_r','RdBu_r','coolwarm','RdGy_r'};
    for i = 1:length(colNames)
        cnt=cnt+1;subplottight(3,3,cnt);
        I = imread(sprintf('/home/mu/horikawa-t/figicon/pycortex_colormap/%s.png',colNames{i}));
        imagesc(I), axis off
        I3 = shiftdim(I(:,idx,:),1);
        fprintf('map = [... %% %s\n',colNames{i})
        for ix = 1:3
            for ixx = 1:64
                fprintf('%d ',I3(ixx,ix))
            end
            fprintf(';...\n')
        end
        fprintf(']''/255;\n')
    end
end