function cond_list = generateCombinations(vars,mode)
% generateCombinations - Generate combinations of conditions.
%
%   cond_list = generateCombinations(vars) generates combinations of
%   conditions specified in the cell array vars. Each element of vars
%   contains a set of conditions. The function returns cond_list, which is a
%   cell array where each element contains a combination of conditions.
%
%   Input:
%       vars - A cell array where each element contains a set of conditions.
%
%   Output:
%       cond_list - A cell array where each element contains a combination of
%                   conditions.
%       mode - a string indicating if the order of output is randomized ('random') or not (default).
%
%   Example:
%       vars = {{'A', 'B'}, {'1', '2', '3'}, {'X', 'Y'}};
%       cond_list = generateCombinations(vars);
%       vars = {{'A', 'B'}, {'1',{ '2', '3'}}, {'X', 'Y'}};
%       cond_list = generateCombinations(vars);
%
%   In the example, cond_list will contain all possible combinations of
%   conditions from vars.
%
%  Written by Tomoyasu Horikawa 20231004
%
%%
% Initialize an empty cell array for cond_list
cond_list = {};
if ~exist('mode','var') || isempty(mode)
    mode = '';
end

% Nested function to generate combinations
    function generateCombinationsRecursive(idx, currentCombination)
        if idx <= numel(vars)
            for j = 1:length(vars{idx})
                newCombination = [currentCombination, vars{idx}(j)];
                % newCombination = [currentCombination, vars{idx}{j}];
                generateCombinationsRecursive(idx + 1, newCombination);
            end
        else
            % Append the currentCombination as a cell element
            cond_list{end + 1} = currentCombination;
        end
    end

% Start generating combinations
generateCombinationsRecursive(1, {});

switch mode
    case 'random'
        cond_list = cond_list(randsample(length(cond_list),length(cond_list)));
end

end