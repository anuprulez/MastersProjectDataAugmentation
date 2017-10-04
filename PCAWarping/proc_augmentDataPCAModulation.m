function novel_epochs = proc_augmentDataPCAModulation(epo_tr_warp, augment_data_size, cnt, ...
    shape, scale, artifact_min_max, reject_artifact, artifact_correction_multiplier, numberOfPCs, clab)
% proc_augmentDataPCAModulation - Augments epochs and remove artifacts from these epochs
%
%  This function augments epochs and then removes artifacts based on a threshold
%
%  Synopsis:
%  proc_augmentDataPCAModulation(epo_tr_warp, augment_data_size, cnt, ...
%      shape, scale, artifact_min_max, reject_artifact, artifact_correction_multiplier, numberOfPCs, clab)
%
%  Arguments:
%  EPO_TR_WARP - A size of training epochs to be used for generating
%  augmented epochs
%  AUGMENT_DATA_SIZE - A size of epochs to be generated
%  CNT - Continuous epoched data
%  SHAPE AND SCALE - Gamma distribution parameters
%  ARTIFACT_MIN_MAX - Theshold for rejecting/accepting an epoch for
%  artifacts
%  REJECT_ARTIFACT - Boolean. True if artifacts to be removed
%  ARTIFACT_CORRECTION_MULTIPLIER - When artifacts are removed, there are many
%  epochs which get rejected. To compensate for this loss of epochs, blow
%  up the original training epochs by this factor and then perform artifact
%  rejection
%  NUMBEROFPCS - Number of principal components to be retained
%  CLAB - Number of original channels
% 
%  Returns:
%  NOVEL_EPOCHS - Artifact corrected novel epochs

% Generate novel epochs by PCA modulation
% Find the multiplier for blowing up the original epochs collection size
% as per the number of novel epochs needed. Also take the number of
% rejected epochs due to the presence of artifacts
updated_augmented_size = artifact_correction_multiplier * augment_data_size;

% Get original epochs in source space
[epochs_PCA, E] = proc_computePCAModulation(epo_tr_warp, cnt, numberOfPCs);
A = E';

disp('Size of epochs in PCA space');
disp(size(epochs_PCA.x));

% Get the epochs and channels size
epochs_channels = size(epochs_PCA.x, 2);
original_epochs_size = size(epochs_PCA.x, 3);

% Loop on till the number of newly generated and artifact corrected epochs 
% matches the number of required augmented epochs
number_augmented_epochs = 0;
while(number_augmented_epochs < augment_data_size)
    % Generate much more number of novel epochs compared to what is
    % required in order to compensate for the loss of epochs due to
    % artifact rejection
    novel_epochs = epochs_PCA;
    novel_epochs.x = zeros(size(epochs_PCA.x,1), epochs_channels, updated_augmented_size);
    novel_epochs.y = zeros(size(epochs_PCA.y,1), updated_augmented_size);

    % Random numbers to pick a random epoch for generating a novel epoch
    random_epochs = randi(original_epochs_size, updated_augmented_size, 1);

    % Collection of random Gamma distributed factors
    modulation_factors = gamrnd(shape,scale,updated_augmented_size,epochs_channels);
    
    % Create novel epochs
    for epoch=1:updated_augmented_size
        % Get random factors from Gamma distribution for all channels
        % of an epoch
        factor = modulation_factors(epoch,:);

        % Randomly choose an original epoch in source space
        random_epoch_number = random_epochs(epoch,1);

        % Create a novel epoch
        for channel=1:epochs_channels
            novel_epochs.x(:,channel,epoch) = epochs_PCA.x(:,channel,random_epoch_number) * factor(1, channel);
        end
        novel_epochs.y(:,epoch) = epochs_PCA.y(:,random_epoch_number);
    end

    % Project back the novel epochs into original space
    novel_epochs_org_space = proc_linearDerivation(novel_epochs, A);
    
    % Add the original channels back to novel epochs
    novel_epochs_org_space.clab = clab;
    
    % Reject artifacts if any in novel epochs in original space
    corrected_novel_epochs_org_space = proc_removeArtifacts(novel_epochs_org_space, artifact_min_max, reject_artifact);

    % Append novel epochs upto the augmented size needed
    try
        augmented_epochs_org_space = proc_appendEpochs(augmented_epochs_org_space, corrected_novel_epochs_org_space);
    catch
        augmented_epochs_org_space = corrected_novel_epochs_org_space;
    end

    % Get the number of augmented epochs in original space generated so far
    number_augmented_epochs = size(augmented_epochs_org_space.x,3);

    disp(['Novel epochs generated: ', num2str(number_augmented_epochs)]);
    disp(['Novel epochs remaining: ', num2str(augment_data_size - number_augmented_epochs)]);

    % Clear the temporary variable
    clear novel_epochs;
end

% Take out the size of novel epochs in original space as required
augmented_epochs_org_space.x = augmented_epochs_org_space.x(:,:,1:augment_data_size);
augmented_epochs_org_space.y = augmented_epochs_org_space.y(:,1:augment_data_size);
novel_epochs = augmented_epochs_org_space;

end

