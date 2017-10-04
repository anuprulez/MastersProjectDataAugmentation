function proc_mainAmplitudeModulation(user_number)
% proc_mainAmplitudeModulation - Entry point of data augmentation by
% amplitude modulation
%
%  This function serves as the entry point to evaluate classification
%  accuracy of data augmentation by amplitude modulation
%
%  Synopsis:
%  proc_mainAmplitudeModulation(user_number)
%
%  Arguments:
%  USER_NUMBER - User's number (VP)
% 
%  Returns:
%  None 

tic;

% Set local paths and initialize BBCI Toolbox
set_localpaths();

% Load data
fs = 100;
epo = proc_loadDataAmplitudeModulation(user_number, fs);

% Augment data by amplitude modulation and evaluate performance
[avg_classfication_orig_size, aug_sizes, original_sizes] = proc_crossValidationAmplitudeModulation(epo);

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


