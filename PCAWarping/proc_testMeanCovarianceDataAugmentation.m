function conditional_accuracy = proc_testMeanCovarianceDataAugmentation(original_tr, augmented_tr, original_te, ival)
%  proc_testMeanCovarianceDataAugmentation - Evaluate different classifiers trained on the combinations of 
%  mean and covariance of original and novel epochs
%
%  This function evaluates different classifiers trained on the combinations of 
%  mean and covariance of original and novel epochs
%
%  Synopsis:
%  proc_testMeanCovarianceDataAugmentation(original_tr, augmented_tr, original_te, ival)
%
%  Arguments:
%  ORIGINAL_TR - Original training epochs
%  AUGMENTED_TR - Augmented (novel) epochs
%  ORIGINAL_TE - Original test epochs
%  IVAL - Time intervals for extracting features
%  
%  Returns:
%  CONDITIONAL_ACCURACY - Accuracies of different classifiers trained on different mean and covariance
%

fv_tr_orig = proc_jumpingMeans(original_tr, ival);
fv_tr_augmented = proc_jumpingMeans(augmented_tr, ival);
fv_te = proc_jumpingMeans(original_te, ival);

%Save all necessary parameter (mean, covariance, inverse of covariance) when training the classifier
classifier_param = {'UsePcov',false,'scaling',true,'StoreMeans',true,'StoreCov',true,'StoreInvcov',true};

% Train clasisfier on original epochs
fv_tr_orig.classifier_param = classifier_param;
C_orig  = trainClassifier(fv_tr_orig, @train_RLDAshrink);
orig_diff = C_orig.mean(:,2)-C_orig.mean(:,1);

% Train classifier on augmented epochs
fv_tr_augmented.classifier_param = classifier_param;
C_augmented  = trainClassifier(fv_tr_augmented, @train_RLDAshrink);
augmented_diff = C_augmented.mean(:,2)-C_augmented.mean(:,1);

% Classifier 1: original mean + original covariance
C1.w = C_orig.invcov*(orig_diff);
C1.b = 0; % Threshold does not matter for AUC

% Classifier 2: augmented covariance + original mean
C2.w = C_augmented.invcov*(orig_diff);
C2.b = 0; 

% Classifier 3: original covariance + augmented mean
C3.w = C_orig.invcov*(augmented_diff);
C3.b = 0; 

% Classifier 4: Augmented mean + augmented covariance
C4.w = C_augmented.invcov*(augmented_diff);
C4.b = 0; 

C = [C1,C2,C3,C4];
conditional_accuracy = zeros(length(C) , 1);
for cond = 1:length(C)           
    out   = applyClassifier(fv_te,C(cond));            
    loss_roc  = mean(loss_rocArea(fv_te.y,out));
    acc = 100*(1-loss_roc);
    fprintf('AUC for cond=%d: %.3f%% \n',cond,acc);
    conditional_accuracy(cond) = acc;
end

end
