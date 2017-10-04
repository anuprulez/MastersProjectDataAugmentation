function [classification_performance, aug_sizes, original_sizes] = proc_crossValidationICAModulation(fs, user_number)
% proc_crossValidationICAModulation - Loads user's data and performs
% cross-validation to asses performance after data augmentation
%
%  This function performs cross-validation for ICA based data augmentation
%  
%  Synopsis:
%  proc_crossValidationICAModulation(FS, USER_NUMBER)
%
%  Arguments:
%  FS - Sampling frequency
%  USER_NUMBER - User's serial number
%
%  Returns:
%  CLASSIFICATION_PERFORMANCE - Classification performances before and
%  after adding augmented epochs
%  AUG_SIZES - Different number of augmented epochs added to original
%  training epochs
%  ORIGINAL_SIZES - Different sizes of original training epochs taken for
%  generating augmented epochs
%

% Set local paths
set_localpaths();
global BTB
warning('off');

% Load user's data
VPs = get_sessionList('session_list_directionstudy'); %20 data sets
VP = VPs{user_number};

% Set parameters
ival_erp = [-200,1200];

% Threshold for rejecting/accepting an epoch
artifact_min_max = 60;
reject_artifact = true;
opt_args={'fs', fs,'runsLoad', 1:6,'critMinMax',artifact_min_max, 'critWhiskerlength', 3, 'classDef', {111:116,101:106; 'Target','Non-target'},...
    'ival', ival_erp,'filt',[0.1 0.5 12 20], 'ibaseline',[-200,0],'supEyeArtifacts',true,'rejectVar',true,'dictn','6D','exp_name','Directionstudy_new'};
[epo, rtrials, cnt, mrk] = custom_readAudioAphasia(VP,opt_args{:});

% Select only scalp channels
epo = proc_selectChannels(epo,util_scalpChannels());
cnt = proc_selectChannels(cnt,util_scalpChannels());
clab = epo.clab;

% N-fold chronological crossvalidation
n_folds = 5;

% Number of runs for k-fold cross-validation
num_runs = 100;

% Mean-covariance conditions
conditional_accuracy_runs = zeros(num_runs, 4);

% Ranges for timesteps to extract features
ival=[100 180; 190 300; 301,450; 450 560; 561 700; 701 850; 851 1000;1001 1200];
[divTr, divTe]= sample_chronKFold(1:length(epo.y), n_folds);

% Sizes of original training epochs taken to create augmented epochs
% These sizes of original training epochs depend on the # of folds of
% cross-validation. For 5-folds, these sizes correspond to 
% [500, 1100, 1800, 2592] respectively of original training epochs
orig_datapoint_increment = [500, 1100, 1800, 2592];
size_orig = size(orig_datapoint_increment, 2);

% Sizes of augmented epochs to be added back to original training epochs
% [0, 100, 200, 400, 800, 1600, 3200, 10000]
aug_datapoint_increment = [0, 100, 200, 400, 800, 1600, 3200, 10000];
size_aug = size(aug_datapoint_increment, 2);
average_classification = zeros(size_aug, 1);

orig_size = zeros(size_orig, 1);
avg_classfication_orig_size = zeros(size_orig, size_aug);

% Gamma distributions parameters
shape = 2;
scale = 1/shape;

artifact_correction_multiplier = 1;
if reject_artifact
    artifact_correction_multiplier = 2;
end

% Variance
classification_runs_variance = zeros(size_orig, size_aug, num_runs);
original_counter = 1;

% Cross-Validation
for orig_idx = 1:numel(orig_datapoint_increment)
    original_data_size = orig_datapoint_increment(orig_idx);
    aug_counter = 1;
    for idx = 1:numel(aug_datapoint_increment)
        avg_classification_runs = zeros(num_runs, 1);
        augment_data_size = aug_datapoint_increment(idx);
        
        % Runs for a particular original epoch size and augmented size
        for run=1:num_runs
            auc_cv = zeros(n_folds, 1);
            conditional_accuracy_folds = zeros(n_folds, 4);
            % Cross validation
            for k=1:n_folds
                disp(['Run number: ' num2str(run)]);
                disp(['Starting Fold number: ' num2str(k)]);
                
                % Epochs for training and test sets
                epo_tr = proc_selectEpochs(epo, divTr{1}{k});
                epo_te = proc_selectEpochs(epo, divTe{1}{k});
                
                % Reject artifacts from test epochs
                disp('Reject artifacts from test epochs');
                epo_te_ICA = proc_removeArtifacts(epo_te, artifact_min_max, reject_artifact);
                original_epo_tr_size = size(epo_tr.x, 3);
                
                disp('Original training epochs');
                disp(original_epo_tr_size);
                
                % Get random epochs from the entire training epochs
                random_idx = randperm(original_epo_tr_size, original_data_size);
                    
                % Select the size of original training epochs to be used as
                % training epochs
                epo_tr_selected_size = proc_selectEpochs(epo_tr, random_idx);
                
                % Include the markings for the correct size of training
                % epochs
                mrk_tr = mrk;
                mrk_tr.time = mrk_tr.time(random_idx);
                mrk_tr.y = mrk_tr.y(:,random_idx);
                
                disp('Original training epochs selected for augmentation');
                disp(size(epo_tr_selected_size.x));
                
                % Reject artifacts in selected size of original training epochs
                training_epochs_corrected = proc_removeArtifacts(epo_tr_selected_size, artifact_min_max, reject_artifact);
                epo_tr_ICA = training_epochs_corrected;
                
                disp('Original training epochs corrected');
                disp(size(epo_tr_ICA.x));

                % Augment data
                if(augment_data_size > 0)
                    
                    % Get ICA augmented epochs
                    augmented_epochs_ICA = proc_augmentDataICAModulation(epo_tr_selected_size, augment_data_size,...
                        mrk_tr, cnt, clab, VP, true, shape, scale, artifact_min_max, reject_artifact, artifact_correction_multiplier);

                    disp('Augmented epochs generated');
                    disp(size(augmented_epochs_ICA.x));

                    conditional_accuracy_folds(k, :) = proc_testMeanCovarianceDataAugmentation(epo_tr_ICA, ...
                                                           augmented_epochs_ICA, epo_te_ICA, ival);
                    
                    % Append augmented epochs to original training epochs
                    epo_tr_ICA = proc_appendEpochs(augmented_epochs_ICA, epo_tr_ICA);
                    
                    disp(['Augmented size after adding: ', num2str(augment_data_size), ' epochs']);
                    disp(size(epo_tr_ICA.x));
                end
                
                disp('Total epochs used for training');
                disp(size(epo_tr_ICA.x));
                
                disp('Epochs used for testing');
                disp(size(epo_te_ICA.x));

                % Features extraction for training and test sets
                fv_tr = proc_jumpingMeans(epo_tr_ICA, ival);
                fv_te = proc_jumpingMeans(epo_te_ICA, ival);

                classifier_param = {'scaling', true, 'StoreMeans', true, 'StoreCov', true, 'StoreInvcov', true};
                fv_tr.classifier_param = classifier_param;

                % Train a classifier
                C  = trainClassifier(fv_tr, @train_RLDAshrink);

                % Apply classifier and get an output
                out = applyClassifier(fv_te, C);

                loss  = mean(loss_rocArea(fv_te.y, out));
                auc_cv(k) = 100 * (1 - loss);
                
                clear epo_tr epo_te epo_tr_ICA epo_te_ICA fv_tr fv_te training_epochs_corrected epo_tr_selected_size augmented_epochs_ICA;
            end
            % Performance of k-th fold after n runs
            avg_classification_runs(run) = mean(auc_cv);
            conditional_accuracy_runs(run, :) = mean(conditional_accuracy_folds, 1);
        end
        classification_runs_variance(original_counter, aug_counter, 1:num_runs) = avg_classification_runs;
        % Performance after adding a number of augmented epochs
        average_classification(aug_counter) = mean(avg_classification_runs);
        fprintf('Classification performance %.2f%% using %d ICA augmented epochs \n', mean(avg_classification_runs), augment_data_size);
        aug_counter = aug_counter + 1;
    end
    % Performance of a particular size of training epochs used for
    % augmentation
    avg_classfication_orig_size(original_counter, 1: size_aug) = average_classification';
    orig_size(original_counter) = original_data_size;
    fprintf('Classification performance using %d original epochs \n', original_data_size);
    original_counter  = original_counter + 1;
end

disp('Mean of mean cov');
disp(mean(conditional_accuracy_runs, 1));

mean_runs_variance = mean(classification_runs_variance, 3);

disp('Mean runs');
disp(mean_runs_variance');

runs_variance = var(classification_runs_variance, 0, 3);
disp('Variance for runs');
disp(runs_variance');

disp('==================');
classification_performance = avg_classfication_orig_size;
aug_sizes = aug_datapoint_increment;
original_sizes = orig_datapoint_increment;


end


