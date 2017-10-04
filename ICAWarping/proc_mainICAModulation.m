function proc_mainICAModulation(user_number)
% proc_mainICAModulation - Entry point to ICA data augmentation strategy
%
%  This function lets one to execute ICA data augmentation strategy for any user (VP) 
%  and see the classification results in a plot
%
%  Synopsis:
%  proc_mainICAModulation(USER_NUMBER)
%  For example, proc_mainICAModulation(5)
%
%  Arguments:
%  USER_NUMBER - Any user's (VP) serial number (1-20)
%
%  Display:
%  Classification results and a plots of classification results
%   

tic;

% Set local paths and initialize BBCI Toolbox
set_localpaths();

% Data augmentation and cross-validation
fs = 100; % Sampling frequency

[avg_classfication_orig_size, aug_sizes, original_sizes] = proc_crossValidationICAModulation(fs, user_number);

% Classification results
disp('Percentages of original training epochs used for augmentation');
disp(original_sizes);

disp('Sizes of augmentated epochs added back to original training epochs');
disp(aug_sizes');

fprintf('Classification performance for user %d \n', user_number);
disp(avg_classfication_orig_size');

disp('Help: Horizontal - Different percentages of original training epochs');
disp('Help: Vertical - Different sizes of augmented epochs added back to original training epochs');

clear;

toc;

end


