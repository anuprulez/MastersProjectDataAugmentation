function warped_signal = proc_oversamplingTimeWarping(shift_values, original_signal, oversampling_factor, oversampled_timesteps)
% proc_oversamplingTimeWarping - Perform time warping in an oversampled epoch
% and then downsample to generate a nove epoch
%
%  This function performs time warping in an oversampled epoch
%  and then downsample it to generate a time warped novel epoch
%
%  Synopsis:
%  proc_oversamplingTimeWarping(shift_values, original_signal, oversampling_factor, oversampled_timesteps)
%
%  Arguments:
%  SHIFT_VALUES - Vector containing shift values for each time step
%  ORIGINAL_SIGNAL - Epoch (singal) to be used for oversampling and time
%  warping
%  OVERSAMPLING_FACTOR - Factor by which an epoch is oversampled from
%  original sampling frequency
%  OVERSAMPLED_TIMESTEPS - A number to which original epoch is oversampled
%  to
%
%  Returns:
%  WARPED_SIGNAL - Novel time warped signal (epoch)
%

% Fill the oversampled matrix with NaN values
channels = size(original_signal, 2);
shifted_signal_oversampled = NaN(oversampled_timesteps, channels);

% Round off the shift values upto 1 precision after decimal
shift_values_rounded = round(shift_values, 1);

% Transfer those values from original matrix to the oversampled matrix
% for which shift values are zero

zero_shift_indexes = find(shift_values_rounded == 0);

% Find indices where shift values are not zero
non_zero_indices = find(shift_values_rounded ~= 0);

% Find the non-zero shift values
non_zero_shift_values = shift_values_rounded(non_zero_indices);

% Find the correct indices for oversampled matrix where values from
% original matrix could fall in
shifted_indices = oversampling_factor * (non_zero_indices - non_zero_shift_values);
shifted_signal_oversampled(oversampling_factor * zero_shift_indexes, :) = original_signal(zero_shift_indexes, :);

% Transfer the values from original matrix to the shifted positions in the
% oversampled matrix
shifted_signal_oversampled(shifted_indices, :) = original_signal(non_zero_indices, :);
shifted_signal_oversampled_channel = shifted_signal_oversampled(1:oversampled_timesteps, :);

% Interpolation
value_pos = find(isnan(shifted_signal_oversampled_channel(:, 1)) == 0);
v = shifted_signal_oversampled_channel(value_pos, :);

% Query points
xq = 1:oversampled_timesteps;

% Interpolated points
vq = interp1(value_pos, v, xq, 'linear', 'extrap');

% Downsample to 1000Hz by taking every 10th sample from 10,000 Hz
shifted_signal_downsampled = downsample(vq, oversampling_factor);

warped_signal = shifted_signal_downsampled;

end


