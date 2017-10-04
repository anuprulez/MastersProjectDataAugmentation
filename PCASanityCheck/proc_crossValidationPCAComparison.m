function proc_crossValidationPCAComparison(fs, user_number)
% proc_crossValidationPCAComparison - Loads user's data and performs
% cross-validation to asses performance after data augmentation
%
%  This function performs cross-validation for ICA based data augmentation
%  
%  Synopsis:
%  proc_crossValidationPCAComparison(FS, USER_NUMBER)
%
%  Arguments:
%  FS - Sampling frequency
%  USER_NUMBER - User's serial number
%
%  Returns:
%  None

% Load user's data
VPs = get_sessionList('session_list_directionstudy'); %20 data sets
VP = VPs{user_number};

% Set parameters
ival_erp = [-200,1200];

opt_args={'fs', fs,'runsLoad', 1:6,'critMinMax',60, 'critWhiskerlength', 3, 'classDef', {111:116,101:106; 'Target','Non-target'},...
    'ival', ival_erp,'filt',[0.1 0.5 12 20], 'ibaseline',[-200,0],'supEyeArtifacts',true,'rejectVar',true,'dictn','6D','exp_name','Directionstudy_new'};
[epo, rtrials, cnt, mrk] = custom_readAudioAphasia(VP,opt_args{:});

% Select only scalp channels
epo = proc_selectChannels(epo,util_scalpChannels());
cnt = proc_selectChannels(cnt,util_scalpChannels());

% N-fold chronological crossvalidation
n_folds = 5;

% Number of principal components
numberOfPCs = 30;

% Ranges for timesteps to extract features
ival=[100 180; 190 300; 301,450; 450 560; 561 700; 701 850; 851 1000;1001 1200];
[divTr, divTe]= sample_chronKFold(1:length(epo.y), n_folds);

% Crossvalidation
auc_cv_Original = zeros(n_folds, 1);
auc_cv_PCA = zeros(n_folds, 1);

for k=1:n_folds
    
    % Epochs for training and test sets
    epo_tr = proc_selectEpochs(epo, divTr{1}{k});
    epo_te = proc_selectEpochs(epo, divTe{1}{k});
    
    % Reject artifacts
    epo_tr = proc_rejectArtifactsMaxMin(epo_tr, 60, 'Clab', {'F*'});
    epo_te = proc_rejectArtifactsMaxMin(epo_te, 60, 'Clab', {'F*'});
    
    % Get classification results for epochs in original space
    auc_cv_Original(k) = proc_evaluateClassificationPerformance(epo_tr, epo_te, ival);
   
    % Move the training and test epochs to PCA space
    epo_tr_PCA = proc_computePCA(epo_tr, cnt, numberOfPCs);
    epo_te_PCA = proc_computePCA(epo_te, cnt, numberOfPCs);
    
    % Get classification results for epochs in PCA space
    auc_cv_PCA(k) = proc_evaluateClassificationPerformance(epo_tr_PCA, epo_te_PCA, ival);
end

disp('Original data performance');
disp(mean(auc_cv_Original));

disp('-------------------------------------');
disp('PCA data performance');
disp(mean(auc_cv_PCA));

end
