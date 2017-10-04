function [classification_performance, aug_sizes, original_sizes] = proc_crossValidationTimeWarping(epo, fs)
% proc_crossValidationTimeWarping - Perform crossvalidation
%
%  This function performs crossvalidation to evaluate classification
%  accuracy using original and augmented training epochs
%
%  Synopsis:
%  proc_crossValidationTimeWarping(epo, fs)
%
%  Arguments:
%  EPO - Complete data (epochs) for a user
%  FS - Sampling frequency
%
%  Returns:
%  CLASSIFICATION_PERFORMANCE - Classification accuracies with different
%  sizes of augmented and original epochs
%  AUG_SIZES - Different number of augmented epochs added to original
%  training epochs
%  ORIGINAL_SIZES - Different sizes of original training epochs taken for
%  generating augmented epochs
%

% Threshold for rejecting/accepting an epoch
artifact_min_max = 60;
reject_artifact = true;
artifact_rejection_multiplier = 1.5;

% Downsampling factor from 1000Hz to 100Hz
downsample_factor = 10;

% N-fold chronological crossvalidation
n_folds = 5;

% Number of runs for k-fold cross-validation
num_runs = 15;

% Mean-covariance conditions
conditional_accuracy_runs = zeros(num_runs, 4);

ival=[100 180; 190 300; 301 450; 450 560; 561 700; 701 850; 851 1000;1001 1200];
[divTr, divTe]= sample_chronKFold(1:length(epo.y), n_folds);

% Sizes of original training epochs taken to create augmented epochs
% [500, 1100, 1800, 2592]
orig_datapoint_increment = [500, 1100, 1800, 2592];
size_orig = size(orig_datapoint_increment, 2);

% Sizes of augmented epochs to be added back to original training epochs
% [0, 100, 200, 400, 800, 1600, 3200, 10000]
aug_datapoint_increment = [0, 100, 200, 400, 800, 1600, 3200, 10000];
size_aug = size(aug_datapoint_increment, 2);
average_classification = zeros(size_aug, 1);

orig_size = zeros(size_orig, 1);
avg_classfication_orig_size = zeros(size_orig, size_aug);

original_counter = 1;

% Variance
classification_runs_variance = zeros(size_orig, size_aug, num_runs);

% Get classification performance of original and augmented epochs using
% k-fold cross-validation
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
                epo_te_TimeWarp = proc_removeArtifacts(epo_te, artifact_min_max, reject_artifact);
                original_epo_tr_size = size(epo_tr.x, 3);
                
                disp('Original training epochs');
                disp(original_epo_tr_size);
                
                % Get random epochs from the entire training epochs
                random_idx = randperm(original_epo_tr_size, original_data_size);
                    
                % Select the size of original training epochs to be used as
                % training epochs
                epo_tr_selected_size = proc_selectEpochs(epo_tr, random_idx);
                
                % Reject artifacts in selected size of original training epochs
                epo_tr_corrected = proc_removeArtifacts(epo_tr_selected_size, artifact_min_max, reject_artifact);
                epo_tr_TimeWarp = epo_tr_corrected;
                
                disp('Original training epochs corrected');
                disp(size(epo_tr_TimeWarp.x));
                
                % Augment epochs
                if(augment_data_size > 0)
                    augmented_epochs_TimeWarp = proc_augmentDataTimeWarping(epo_tr_selected_size, augment_data_size, fs, ...
                        artifact_rejection_multiplier, artifact_min_max, reject_artifact);
                    
                    disp('Augmented epochs generated');
                    disp(size(augmented_epochs_TimeWarp.x));

                    conditional_accuracy_folds(k, :) = proc_testMeanCovarianceDataAugmentation(epo_tr_TimeWarp, ...
                                                           augmented_epochs_TimeWarp, epo_te_TimeWarp, ival);
                    
                    epo_tr_TimeWarp = proc_appendEpochs(epo_tr_TimeWarp, augmented_epochs_TimeWarp);
                    
                    disp(['Augmented size after adding: ', num2str(augment_data_size), ' epochs']);
                    disp(size(epo_tr_TimeWarp.x));
                    
                end
                
                % Downsample the train and test epochs from 1000Hz to 100Hz
                epo_tr_TimeWarp.x = downsample(epo_tr_TimeWarp.x, downsample_factor);
                epo_te_TimeWarp.x = downsample(epo_te_TimeWarp.x, downsample_factor);
                
                % Update the sampling frequency
                epo_tr_TimeWarp.fs = 100;
                epo_te_TimeWarp.fs = 100;

                % Update the 't' variable of the structs
                epo_tr_TimeWarp.t = -190:10:1200;
                epo_te_TimeWarp.t = -190:10:1200;
                
                disp('Total epochs used for training');
                disp(size(epo_tr_TimeWarp.x));
                
                disp('Epochs used for testing');
                disp(size(epo_te_TimeWarp.x));

                % Features extraction for training and test sets
                fv_tr = proc_jumpingMeans(epo_tr_TimeWarp, ival);
                fv_te = proc_jumpingMeans(epo_te_TimeWarp, ival);

                classifier_param = {'scaling', true, 'StoreMeans', true, 'StoreCov', true, 'StoreInvcov', true};
                fv_tr.classifier_param = classifier_param;

                % Classifier training
                C  = trainClassifier(fv_tr, @train_RLDAshrink);
               
                % Classifier output
                out = applyClassifier(fv_te, C);
                loss  = mean(loss_rocArea(fv_te.y, out));
                auc_cv(k) = 100 * (1 - loss);
                clear epo_tr epo_te epo_tr_TimeWarp epo_te_TimeWarp fv_tr fv_te epo_tr_selected_size epo_tr_corrected augmented_epochs_TimeWarp;
            end
            avg_classification_runs(run) = mean(auc_cv);
            conditional_accuracy_runs(run, :) = mean(conditional_accuracy_folds, 1);
        end
        classification_runs_variance(original_counter, aug_counter, 1:num_runs) = avg_classification_runs;
        % Performance after adding a number of augmented epochs
        average_classification(aug_counter) = mean(avg_classification_runs);
        fprintf('Classification performance %.2f%% using %d Time warped epochs \n', mean(avg_classification_runs), augment_data_size);
        aug_counter = aug_counter + 1;
    end
    % Performance of a particular size of training epochs used for augmentation
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

