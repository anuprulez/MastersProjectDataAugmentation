function proc_mainPCAComparison(user_number)
% proc_mainPCAComparison - Entry point to PCA projection sanity check
%
%  This function lets one to compare the classification accuracy of epochs in
%  original as well as PCA space
%
%  Synopsis:
%  proc_mainPCAComparison(USER_NUMBER)
%  For example, proc_mainPCAComparison(1)
%
%  Arguments:
%  USER_NUMBER - Any user's (VP) serial number (1-10)
%
%  Returns:
%  None
%   

tic;

% Set local paths and initialize BBCI Toolbox
set_localpaths();
global BTB
warning('off');

fs = 100; % Sampling frequency
proc_crossValidationPCAComparison(fs, user_number);

toc;

end