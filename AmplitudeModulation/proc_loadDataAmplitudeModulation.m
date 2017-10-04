function epo = proc_loadDataAmplitudeModulation(user_number, fs)
%  proc_loadDataAmplitudeModulation - Loads user's data at the specified
%  sampling frequency and rejects artifacts
%
%  This function loads user's data and rejects artifacts
%
%  Synopsis:
%  proc_loadDataAmplitudeModulation(user_number, fs)
%  
%  Arguments:
%  USER_NUMBER - User's number (VP)
%  FS - Sampling frequency
%
%  Returns:
%  EPO - Complete data for any user

warning('off');
disp('data loading started...');

% Get data for the specified user
VPs = get_sessionList('session_list_directionstudy'); %20 data sets
VP =VPs{user_number};

opt_args={'fs', fs,'runsLoad', 1:6,'critMinMax',60, 'critWhiskerlength', 3, 'classDef', {111:116,101:106; 'Target','Non-target'},...
    'ival', [-200,1200],'filt',[0.1 0.5 12 20], 'ibaseline',[-200,0],'supEyeArtifacts',true,'rejectVar',true,'dictn','6D','exp_name','Directionstudy_new'};
[epo, rtrials, cnt, mrk] = custom_readAudioAphasia(VP,opt_args{:});

% Use only scalp channels
epo = proc_selectChannels(epo,util_scalpChannels());

disp('data loaded');

end



