function [epo_source, ICA] = proc_computeICAModulation(original_epo_tr, mrk_tr, cnt, VP)
% proc_computeICAModulation - Projects epochs into source space
% 
%  This function projects epochs into source space
%
%  Synopsis:
%  proc_computeICAModulation(original_epo_tr, mrk_tr, cnt, VP)
%
%  Arguments:
%  ORIGINAL_EPO_TR - A selected size of training epochs to be used for projecting into source
%  space
%  MRK_TR - Markings of the selected size of training epochs
%  CNT - Continuous epoched data
%  VP - User
%
%  Returns:
%  EPO_SOURCE - Epochs in the source space
%  ICA - Object containing mixing and unmixing matrices
%

% Keep 30 components out of 63 components (= number of channels)
numberOfICs = 30;

% Set the ICA parameters
ival_erp = [-200,1200];
opt_ica  = {'useLoadedICs',false,'rmvInterBlockSignals', true,'varianceAR',false,...
'ival',ival_erp,'high_pass_fs',2,'critWhiskerlength', 2,'critWhiskerperc', 10,...
'MARA', false,'numberOfICs', numberOfICs,'useFullRank', false,'maxNumIterations', 300,...
'preComputeICA', true,'save_data', false};

% Compute the ICA components
[ ~, ICA, ~ ]=proc_ICAArtifactRejection(cnt,VP,mrk_tr,opt_ica{:});

% Remove field 'ICs' to free memory
ICA=rmfield(ICA,'ICs');

% Move the epochs into source space - 
% the new channels are the independent components
epo_source=proc_linearDerivation(original_epo_tr,ICA.W');

end


