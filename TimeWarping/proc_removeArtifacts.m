function correct_epochs =  proc_removeArtifacts(epochs, artifact_min_max, reject_artifact)
% proc_removeArtifacts - Remove artifacts from epochs
%
%  This function removes artifacts based on a threshold
%
%  Synopsis:
%  proc_removeArtifacts(EPOCHS, ARTIFACT_MIN_MAX, REJECT_ARTIFACT)
%
%  Arguments:
%  EPOCHS - All Epochs
%  ARTIFACT_MIN_MAX - Threshold to decide whether to retain an epoch
%  REJECT_ARTIFACT - Boolean. True if needed to reject artifactual signals
%
%  Returns:
%  CORRECT_EPOCHS - Artifact corrected epochs

% If set to true, reject artifacts and return corrected epochs
% otherwise return the same epochs
if reject_artifact
    if isfield(epochs, 'event')
        epochs = rmfield(epochs,'event');
    end
    disp('Epochs before artifact rejection');
    disp(size(epochs.x));
    
    % Remove artifacts only from frontal channels
    correct_epochs = proc_rejectArtifactsMaxMin(epochs, artifact_min_max, 'Clab', {'F*'});
    
    disp('Epochs after artifact rejection');
    disp(size(correct_epochs.x));
else
    correct_epochs = epochs;
end
