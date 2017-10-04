function classification_result = proc_evaluateClassificationPerformance(training_epo, test_epo, ival)
% proc_evaluateClassificationPerformance - Evaluate classification
% performance
% 
%  This function evaluates classification performance on test epochs after
%  training the classifier on training epochs
%
%  Synopsis:
%  proc_evaluateClassificationPerformance(TRAINING_EPO, TEST_EPO, IVAL)
%
%  Arguments:
%  TRAINING_EPO - Training epochs
%  TEST_EPO - Test epochs
%  IVAL - Time intervals (in ms) for extracting features time intervals for
%  extracting features
%
%  Returns:
%  CLASSIFICATION_RESULT - Classification result (in percentage)
%

% Features extraction for training and test sets
fv_tr = proc_jumpingMeans(training_epo, ival);
fv_te = proc_jumpingMeans(test_epo, ival);

classifier_param = {'scaling', true, 'StoreMeans', true, 'StoreCov', true, 'StoreInvcov', true};
fv_tr.classifier_param = classifier_param;

% Train a classifier
C  = trainClassifier(fv_tr, @train_RLDAshrink);

% Apply classifier and get an output
out = applyClassifier(fv_te, C);

loss  = mean(loss_rocArea(fv_te.y, out));
classification_result = 100 * (1 - loss);

end