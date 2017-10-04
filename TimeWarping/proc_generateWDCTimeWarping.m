function warped_epochs = proc_generateWDCTimeWarping(epochs, fs)
% proc_generateWDCTimeWarping - Generate augmented (time warped) signals
%
%  This function generates novel time warped signals using a set of original signals by generating a warp
%  defining curve and taking out time shift values from this curve
%
%  Synopsis:
%  proc_generateWDCTimeWarping(epochs, fs)
%
%  Arguments:
%  EPOCHS - Epochs to be used for generating augmented epochs
%  FS - Sampling frequency
%
%  Returns:
%  WARPED_EPOCHS - Augmented epochs
%

section_size =  1.4 * fs;
epoch_size =  1.4 * fs;
max_shift = 0.2 * fs; % Allowable value for maximum shift
oversampled_timesteps = 14000;
starting_offset = 70000;
low_freq = 0.2;
high_freq = 0.5;
filter_order = 8;
number_epochs = size(epochs, 3);
sampling_factor = oversampled_timesteps / epoch_size;
warped_epochs = zeros(size(epochs, 1), size(epochs, 2), number_epochs);

%% Generate white noise
samples = starting_offset + number_epochs * section_size;
w1 = randn(1, samples);

%% Filter signal from white noise using low pass filter
bpFilt = designfilt('bandpassiir', 'FilterOrder', filter_order, 'HalfPowerFrequency1', low_freq, ...
   'HalfPowerFrequency2', high_freq, 'SampleRate', fs);
filtered_curve = filter(bpFilt, w1);

%% Linear function
% Generate a linear function for scaling 
% the filtered curve to generate warp defining curve
lf_x = 1:section_size;
lf_slope = 1/6; % (200 - 0) / (1200 - 0)
lf_y = zeros(1, section_size);
lf_y(1, 1:epoch_size - max_shift) = lf_slope * lf_x(1, 1:epoch_size - max_shift);
lf_y(1, epoch_size - max_shift + 1: epoch_size) = max_shift;

% Generate time warped epochs
for epoch_number=1:number_epochs
    epoch = epochs(:, :, epoch_number);

    % Take out a section for an epoch from the filtered signal
    start_pos = (epoch_number - 1) * section_size + starting_offset;
    end_pos = start_pos + epoch_size;
    section = filtered_curve(1, start_pos + 1: end_pos);

    %% Scale the section between 1 and -1
    min_y = min(section);
    max_y = max(section);
    section = (2*(section - min_y)/(max_y - min_y))-1;
    
    %% Generate warp defining curve
    warp_defining_curve = (section .* lf_y);
    warped_epochs(:, :, epoch_number) = proc_oversamplingTimeWarping(warp_defining_curve, epoch, sampling_factor, oversampled_timesteps);
end

end


