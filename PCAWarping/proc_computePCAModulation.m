function [epo_PCA_space, E] = proc_computePCAModulation(epochs, cnt, numberOfPCs)
% proc_computePCAModulation - Projects epochs into source space
% 
%  This function projects epochs into source space
%
%  Synopsis:
%  proc_computePCAModulation(epochs, cnt, numberOfPCs)
%
%  Arguments:
%  EPOCHS - A size of epochs in original space to be projected into PCA
%  space
%  CNT - Continuous epoched data
%  NUMBEROFPCS - Number of principal components to be retained 
%
%  Returns:
%  EPO_PCA_SPACE - Epochs in the source space
%  E - Eigen vector matrix
%

[E,D] = custom_fastica(cnt.x','lastEig', numberOfPCs, 'only', 'pca');

% Move the epochs to PCA space
epo_PCA_space=proc_linearDerivation(epochs, E);

end


