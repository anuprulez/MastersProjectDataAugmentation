function novel_corrected_epochs = proc_augmentDataTimeWarping(original_epochs, aug_size, fs, ...
    artifact_rejection_multiplier, artifact_min_max, reject_artifact)
% proc_augmentDataTimeWarping - Create augmented epochs of required size
%
%  This function generates novel epochs of required size
%
%  Synopsis:
%   proc_augmentDataTimeWarping(original_epochs, aug_size, fs, ...
%       artifact_rejection_multiplier, artifact_min_max, reject_artifact)
%  Arguments:
%  ORIGINAL_EPOCHS - A size of training epochs to be used for generating
%  augmented epochs
%  AUG_SIZE - A size of epochs to be generated
%  FS - Sampling frequency
%  ARTIFACT_REJECTION_MULTIPLIER - Inflate the size of augmented epochs
%  wanted to compensate for the loss of epochs due to artifacts
%  ARTIFACT_MIN_MAX - Threshold for min and max difference to accept an
%  epoch
%  REJECT_ARTIFACT - Reject artifact if true otherwise not
%
%  Returns:
%  WARPED_EPOCHS - Artifact corrected novel epochs
%

warped_epochs = original_epochs;
updated_aug_size = aug_size * artifact_rejection_multiplier;
number_epochs = size(original_epochs.x, 3);
random_aug_epochs = randi(number_epochs, updated_aug_size, 1);

wanted_epochs = original_epochs.x(:,:,random_aug_epochs);
label = original_epochs.y(:, random_aug_epochs);

size_augmented = 0;
while(size_augmented < aug_size)
    % Get time warped epochs
    warped_data = proc_generateWDCTimeWarping(wanted_epochs, fs);
    warped_epochs.x = warped_data;
    warped_epochs.y = label;
    warped_epochs_corrected = proc_removeArtifacts(warped_epochs, artifact_min_max, reject_artifact);
    try
        warped_epochs_corrected_aug_size = proc_appendEpochs(warped_epochs_corrected, warped_epochs_corrected_aug_size);
    catch
        warped_epochs_corrected_aug_size = warped_epochs_corrected;
    end
    size_augmented = size(warped_epochs_corrected_aug_size.x, 3);
    disp(['Novel epochs generated: ', num2str(size_augmented)]);
    disp(['Novel epochs remaining: ', num2str(aug_size - size_augmented)]);
end
novel_corrected_epochs = proc_selectEpochs(warped_epochs_corrected_aug_size, 1:aug_size);
end

