function epo_PCA_space = proc_computePCA(epo, cnt, numberOfPCs)
% proc_computePCA - Project epochs into PCA space
% 
%  This function projects epochs into PCA space having "numberOfPCs" many
%  components
%
%  Synopsis:
%  proc_computePCA(EPO, CNT, NUMBEROFPCS)
%
%  Arguments:
%  EPO - Epochs to be projected into PCA space
%  space
% CNT - Continuous epochs
% NUMBEROFPCS - Number of Principal Components to be retained out of 63
%
%  Returns:
%  EPO_PCA_SPACE - Epochs in PCA space
%

% Compute PCA components. Get Eigen vectors (E) and Eigen values (D)
[E,D] = custom_fastica(cnt.x','lastEig', numberOfPCs, 'only', 'pca');

% Epochs in PCA space
fprintf('\nApplying PCA projection \n');
epo_PCA_space=proc_linearDerivation(epo, E);

disp(size(epo_PCA_space.x));

end
