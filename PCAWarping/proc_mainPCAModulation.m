function proc_mainPCAModulation(user_number)
% proc_mainPCAModulation - Entry point to ICA data augmentation strategy
%
%  This function lets one to execute ICA data augmentation strategy for any user (VP) 
%  and see the classification results in a plot
%
%  Synopsis:
%  proc_mainPCAModulation(USER_NUMBER)
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
warning('off');

% Data augmentation and cross-validation
fs = 100; % Sampling frequency

[avg_classfication_orig_size, aug_sizes, original_sizes] = proc_crossValidationPCAModulation(fs, user_number);

% Classification results
Original_Sizes = original_sizes;
Aug_Sizes = aug_sizes;

disp('Original training epochs used for augmentation');
disp(Original_Sizes);

disp('Sizes of augmented epochs added back to original training epochs');
disp(Aug_Sizes');

fprintf('Classification performance for user %d using PCA augmentation \n', user_number);
disp('---------------------------------------------------------');
disp(avg_classfication_orig_size');
disp('---------------------------------------------------------');

disp('Help: Horizontal - Different percentages of original training epochs');
disp('Help: Vertical - Different sizes of augmented epochs added back to original training epochs');

clear;

toc;

end

