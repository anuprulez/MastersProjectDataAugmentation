function novel_epochs = proc_warpEpochAmplitudeModulation(original_epochs, wf_base, wf_diff, augment_data_size)
%  proc_warpEpochAmplitudeModulation - Generate novel epochs of required
%  size using warp factors following a uniform distribution by modulating the amplitudes
%
%  This function generates novel epochs by multiplying a random warp factor
%  to original epochs
%
%  Synopsis:
%  proc_warpEpochAmplitudeModulation(original_epochs, wf_base, wf_diff, augmented_data_size)
%
%  Arguments:
%  ORIGINAL_EPOCHS - A size of original training epochs to be used for generating
%  augmented epochs
%  WF_BASE and WF_DIFF - Defines a range from which random warp factors are
%  generated following a uniform distribution
%  AUGMENTED_DATA_SIZE - Number of novel epochs needed
%
%  Returns:
%  NOVEL EPOCHS - Novel epochs
%

novel_epochs = original_epochs;
warped_data = zeros(size(original_epochs.x, 1), size(original_epochs.x, 2), augment_data_size);
label = zeros(size(original_epochs.y, 1), augment_data_size);

% Generate random warp factors with mean close to 1
warp_factors =  wf_base + wf_diff * rand(augment_data_size,1);
disp(['Mean warp factors: ', num2str(mean(warp_factors))]);

% Generate random epochs numbers corresponding to the size of novel epochs
% wanted
random_epoch_number = randi(size(original_epochs.x, 3), augment_data_size, 1);

for epoch=1:augment_data_size
    % Get the warping factor
    factor = warp_factors(epoch,1);
    
    % Get random epoch number
    epoch_number = random_epoch_number(epoch, 1);
    
    % Create a novel epoch
    warped_data(:,:, epoch) =  original_epochs.x(:,:, epoch_number) * factor;
    label(:, epoch) = original_epochs.y(:, epoch_number);
end
novel_epochs.x = warped_data;
novel_epochs.y = label;
end

