function proc_mainTimeWarping(user_number)
% proc_mainTimeWarping - Entry point to data augmentation by time warping
% strategy
%
%  This function serves as the entry point to evaluate classification
%  performance using time warping data augmentation
%
%  Synopsis:
%  proc_mainTimeWarping(user_number)
%
%  Arguments:
%  USER_NUMBER -User's number (VP)
%
%  Returns:
%  None
%

tic;

% Set local paths and initialize BBCI Toolbox
set_localpaths();

% Load data & remove artefacts
fs = 1000;
epo = proc_loadDataTimeWarping(user_number, fs);

% Augment data by time warping and evaluate performance using cross-validation
[avg_classfication_orig_size, aug_sizes, original_sizes] = proc_crossValidationTimeWarping(epo, fs);

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


