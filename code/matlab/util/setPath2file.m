function [filename,filepath] = setPath2file(filename)
% function [filename,filepath] = setPath2file(filename)
% 
% this function make directories through tha path to the file
% 
% [Input]
%  -filename: filename
% 
% [Output]
%  -filename: filename
%  -filepath: path to the file
% 
% Written by Tomoyasu Horikawa 20231003
%
filepath = setdir(fileparts(filename));

