%% Setup workspace
%clear;
cd /data/nil-bluearc/ances/millarp/fc_projects/analyses/;

%% Load Datasets - volumetrics
abneg_train_vol = load('reg_learner_data_volharm_abnegtrain_487_220506.mat').vol_features;
abneg_test_vol = load('reg_learner_data_volharm_abnegtest_145_220506.mat').vol_features;
abposs_vol = load('reg_learner_data_volharm_abpos_155_220506.mat').vol_features;
cdr051s_vol = load('reg_learner_data_volharm_cdr_182_220506.mat').vol_features;
%% Load FC & labels
abneg_train_fc = load('reg_learner_data_fc300harm_abnegtrain_489_211203.mat').abneg_train;%.fcharm_features;
abneg_test_fc = load('reg_learner_data_fc300harm_abnegtest_145_211203.mat').abneg_test;%.fcharm_features;
abposs_fc = load('reg_learner_data_fc300harm_abpos_155_211203.mat').abposs;%.fcharm_features;
cdr051s_fc = load('reg_learner_data_fc300harm_cdr_183_211203.mat').cdr051s;%.fcharm_features;

abneg_train_fcnet = load('reg_learner_data_fc17harm_abnegtrain_489_211215.mat').abneg_train;%.fcharm_features;
abneg_test_fcnet = load('reg_learner_data_fc17harm_abnegtest_145_211215.mat').abneg_test;%.fcharm_features;
abposs_fcnet = load('reg_learner_data_fc17harm_abpos_155_211215.mat').abposs;%.fcharm_features;
cdr051s_fcnet = load('reg_learner_data_fc17harm_cdr_183_211215.mat').cdr051s;%.fcharm_features;

abneg_train_labels = load('fc17harm_labels_abnegtrain_489_211203').abneg_train_labels;%.fcharm_labels;
abneg_test_labels = load('fc17harm_labels_abnegtest_145_211203').abneg_test_labels;%.fcharm_labels;
abposs_labels = load('fc17harm_labels_abpos_155_211203').abpos_labels;%.fcharm_labels;
cdr051s_labels = load('fc17harm_labels_cdr_183_211203').cdr051_labels;%.fcharm_labels;

%% Exclude FC subs w/ missing Vol & merge across modalities
%abneg_train
abneg_train_fc = abneg_train_fc(abneg_train_labels.vol_check > 0, :);
abneg_train_labels = abneg_train_labels(abneg_train_labels.vol_check > 0, :);
abneg_train_fc.SessionID = abneg_train_labels.SessionID;
abneg_train_fc = sortrows(abneg_train_fc, 'SessionID');

abneg_train_fcnet = abneg_train_fcnet(abneg_train_labels.vol_check > 0, :);
abneg_train_labels = abneg_train_labels(abneg_train_labels.vol_check > 0, :);
abneg_train_fcnet.SessionID = abneg_train_labels.SessionID;
abneg_train_fcnet = sortrows(abneg_train_fcnet, 'SessionID');

abneg_train_labels = sortrows(abneg_train_labels, 'SessionID');
abneg_train_vol = sortrows(abneg_train_vol, 'SessionID');
% plot 
figure;
scatter(abneg_train_vol.Age, abneg_train_fc.Age);
title('abneg train');
abneg_train = [abneg_train_fc(:,44851), abneg_train_vol(:,17:117), abneg_train_fc(:,1:44850), abneg_train_fcnet(:,2:154)];
%abneg_train = [abneg_train_fc(:,155), abneg_train_vol(:,17:117), abneg_train_fc(:,2:154)];

%abneg_test
abneg_test_fc = abneg_test_fc(abneg_test_labels.vol_check > 0, :);
abneg_test_labels = abneg_test_labels(abneg_test_labels.vol_check > 0, :);
abneg_test_fc.SessionID = abneg_test_labels.SessionID;
abneg_test_fc = sortrows(abneg_test_fc, 'SessionID');

abneg_test_fcnet = abneg_test_fcnet(abneg_test_labels.vol_check > 0, :);
abneg_test_labels = abneg_test_labels(abneg_test_labels.vol_check > 0, :);
abneg_test_fcnet.SessionID = abneg_test_labels.SessionID;
abneg_test_fcnet = sortrows(abneg_test_fcnet, 'SessionID');

abneg_test_labels = sortrows(abneg_test_labels, 'SessionID');
abneg_test_vol = sortrows(abneg_test_vol, 'SessionID');
% plot 
figure;
scatter(abneg_test_vol.Age, abneg_test_fc.Age);
title('abneg_test');
abneg_test = [abneg_test_fc(:,44851), abneg_test_vol(:,17:117), abneg_test_fc(:,1:44850), abneg_test_fcnet(:,2:154)];
%abneg_test = [abneg_test_fc(:,155), abneg_test_vol(:,17:117), abneg_test_fc(:,2:154)];

%abposs
abposs_fc = abposs_fc(abposs_labels.vol_check > 0, :);
abposs_labels = abposs_labels(abposs_labels.vol_check > 0, :);
abposs_fc.SessionID = abposs_labels.SessionID;
abposs_fc = sortrows(abposs_fc, 'SessionID');

abposs_fcnet = abposs_fcnet(abposs_labels.vol_check > 0, :);
abposs_labels = abposs_labels(abposs_labels.vol_check > 0, :);
abposs_fcnet.SessionID = abposs_labels.SessionID;
abposs_fcnet = sortrows(abposs_fcnet, 'SessionID');

abposs_labels = sortrows(abposs_labels, 'SessionID');
abposs_vol = sortrows(abposs_vol, 'SessionID');
% plot 
figure;
scatter(abposs_vol.Age, abposs_fc.Age);
title('abposs');
abposs = [abposs_fc(:,44851), abposs_vol(:,17:117), abposs_fc(:,1:44850), abposs_fcnet(:,2:154)];
%abposs = [abposs_fc(:,155), abposs_vol(:,17:117), abposs_fc(:,2:154)];

%cdr051s
cdr051s_fc = cdr051s_fc(cdr051s_labels.vol_check > 0, :);
cdr051s_labels = cdr051s_labels(cdr051s_labels.vol_check > 0, :);
cdr051s_fc.SessionID = cdr051s_labels.SessionID;
cdr051s_fc = sortrows(cdr051s_fc, 'SessionID');

cdr051s_fcnet = cdr051s_fcnet(cdr051s_labels.vol_check > 0, :);
cdr051s_labels = cdr051s_labels(cdr051s_labels.vol_check > 0, :);
cdr051s_fcnet.SessionID = cdr051s_labels.SessionID;
cdr051s_fcnet = sortrows(cdr051s_fcnet, 'SessionID');

cdr051s_labels = sortrows(cdr051s_labels, 'SessionID');
cdr051s_vol = sortrows(cdr051s_vol, 'SessionID');
% plot 
figure;
scatter(cdr051s_vol.Age, cdr051s_fc.Age);
title('cdr051s');
cdr051s = [cdr051s_fc(:,44851), cdr051s_vol(:,17:117), cdr051s_fc(:,1:44850), cdr051s_fcnet(:,2:154)];
%cdr051s = [cdr051s_fc(:,155), cdr051s_vol(:,17:117), cdr051s_fc(:,2:154)];

%% Merge abnegs then split into full-age-range test and validation sample (80/20)
abnegs = [abneg_train];
abneg_labels = [abneg_train_labels];

[m,n] = size(abnegs) ;
P = 0.8 ;
rng(1106);
idx = randperm(m)  ;
abneg_train = abnegs(idx(1:round(P*m)),:) ; 
abneg_val = abnegs(idx(round(P*m)+1:end),:) ;
abneg_train_labels = abneg_labels(idx(1:round(P*m)),:) ; 
abneg_val_labels = abneg_labels(idx(round(P*m)+1:end),:) ;

%% LOOP MODEL TRAINING FOR VOLUMETRIC, FC, AND COMBINED MODELS
i = 1;
nsims = 1000;

train_RMSE_fc = zeros(nsims, 1);
train_MAE_fc = zeros(nsims, 1);
train_R2_fc = zeros(nsims, 1);

train_RMSE_fcnet = zeros(nsims, 1);
train_MAE_fcnet = zeros(nsims, 1);
train_R2_fcnet = zeros(nsims, 1);

train_RMSE_vol = zeros(nsims, 1);
train_MAE_vol = zeros(nsims, 1);
train_R2_vol = zeros(nsims, 1);

train_RMSE_volsub = zeros(nsims, 1);
train_MAE_volsub = zeros(nsims, 1);
train_R2_volsub = zeros(nsims, 1);

train_RMSE_volct = zeros(nsims, 1);
train_MAE_volct = zeros(nsims, 1);
train_R2_volct = zeros(nsims, 1);

%train_RMSE_volfc = zeros(nsims, 1);
%train_MAE_volfc = zeros(nsims, 1);
%train_R2_volfc = zeros(nsims, 1);

train_RMSE_volfcstack_gpr = zeros(nsims, 1);
train_MAE_volfcstack_gpr = zeros(nsims, 1);
train_R2_volfcstack_gpr = zeros(nsims, 1);

train_RMSE_volfcstack2_gpr = zeros(nsims, 1);
train_MAE_volfcstack2_gpr = zeros(nsims, 1);
train_R2_volfcstack2_gpr = zeros(nsims, 1);

train_RMSE_volfcstack_tree = zeros(nsims, 1);
train_MAE_volfcstack_tree = zeros(nsims, 1);
train_R2_volfcstack_tree = zeros(nsims, 1);

train_RMSE_volfcstack_rf = zeros(nsims, 1);
train_MAE_volfcstack_rf = zeros(nsims, 1);
train_R2_volfcstack_rf = zeros(nsims, 1);

train_RMSE_volfcstack_svm = zeros(nsims, 1);
train_MAE_volfcstack_svm = zeros(nsims, 1);
train_R2_volfcstack_svm = zeros(nsims, 1);

train_RMSE_volfcstack_reg = zeros(nsims, 1);
train_MAE_volfcstack_reg = zeros(nsims, 1);
train_R2_volfcstack_reg = zeros(nsims, 1);

% Merge abnegs the split into full-age-range test and validation sample (80/20)
abnegs = [abneg_train; abneg_val];
abneg_labels = [abneg_train_labels; abneg_val_labels];

abnegs_norm = abnegs;
[nobs, nfeats] = size(abnegs);

% Normalize features in full training set
for f = 2:nfeats
    abnegs_norm(:,f) = normalize(abnegs(:,f));

end

abnegs_vol = abnegs(:,1:102);
abnegs_norm_vol = abnegs_norm(:,1:102);
%%

tic

parfor i = 1:nsims
    %%
    [m,n] = size(abnegs_norm) ;
    P = 0.8 ;
    rng(i);
    %rng(1106);
    idx = randperm(m)  ;
    abneg_train_i = abnegs_norm(idx(1:round(P*m)),:) ; 
    abneg_test_i = abnegs_norm(idx(round(P*m)+1:end),:) ;
    abneg_train_labels_i = abneg_labels(idx(1:round(P*m)),:) ; 
    abneg_test_labels_i = abneg_labels(idx(round(P*m)+1:end),:) ;
    %%
    %VOL
    %% Optimize and save model
%     rng(1106);
%     [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_volthick_111_opt(abneg_train);
%     %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_vol101_age(abneg_train_i);
%     save('train_gpr_rq_volthick_111_opt_model_390_220506', 'trainedModel');
%     save('train_gpr_rq_volthick_111_opt_pred_390_220506', 'validationPredictions');
    %%
    rng(1106);
    [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_volthick_111(abneg_train_i);

    % Extract ABNeg age predictions & true ages
    %abneg_age_pred = trained_gpr.predictFcn(abneg_train(:, 1:44253));
    %abneg_age_pred_train = validationPredictions;
    %abneg_age_true_train = abneg_train_i.Age;
    
    abneg_age_pred_train = trainedModel.predictFcn(abneg_test_i);%(:,1:105));%(:,1:44253));
    abneg_age_true_train = abneg_test_i.Age;

    % Validation R2 & MAE
    validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

    cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
    validationR2 = cormat(1,2)^2;

    train_RMSE_vol(i) = validationRMSE;
    train_MAE_vol(i) = validationMAE;
    train_R2_vol(i) = validationR2;
    predictions_vol_i_train = validationPredictions;
    predictions_vol_i_test = abneg_age_pred_train;
    
    %%
    %volsub SUB
%     %% Optimize and save model
%     rng(1106);
%     [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_volsub_33_opt(abneg_train);
%     %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_volsub101_age(abneg_train_i);
%     save('train_gpr_rq_volsub_33_opt_model_390_220506', 'trainedModel');
%     save('train_gpr_rq_volsub_33_opt_pred_390_220506', 'validationPredictions');
%     %%

     rng(1106);
     [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_volsub_33(abneg_train_i);


    % Extract ABNeg age predictions & true ages
    %abneg_age_pred = trained_gpr.predictFcn(abneg_train(:, 1:44253));
    %abneg_age_pred_train = validationPredictions;
    %abneg_age_true_train = abneg_train_i.Age;
    
    abneg_age_pred_train = trainedModel.predictFcn(abneg_test_i);%(:,1:105));%(:,1:44253));
    abneg_age_true_train = abneg_test_i.Age;

    % Validation R2 & MAE
    validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

    cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
    validationR2 = cormat(1,2)^2;

    train_RMSE_volsub(i) = validationRMSE;
    train_MAE_volsub(i) = validationMAE;
    train_R2_volsub(i) = validationR2;
    predictions_volsub_i_train = validationPredictions;
    predictions_volsub_i_test = abneg_age_pred_train;
    
    %%
    %volct
%     %% Optimize and save model
%     rng(1106);
%     
%     [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_volct_68_opt(abneg_train);
%     %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_volct101_age(abneg_train_i);
%     save('train_gpr_rq_volct_68_opt_model_390_220506', 'trainedModel');
%     save('train_gpr_rq_volct_68_opt_pred_390_220506', 'validationPredictions');
%     %%


     rng(1106);
     
     [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_volct_68(abneg_train_i);
    % Extract ABNeg age predictions & true ages
    %abneg_age_pred = trained_gpr.predictFcn(abneg_train(:, 1:44253));
    %abneg_age_pred_train = validationPredictions;
    %abneg_age_true_train = abneg_train_i.Age;
    
    abneg_age_pred_train = trainedModel.predictFcn(abneg_test_i);%(:,1:105));%(:,1:44253));
    abneg_age_true_train = abneg_test_i.Age;

    % Validation R2 & MAE
    validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

    cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
    validationR2 = cormat(1,2)^2;

    train_RMSE_volct(i) = validationRMSE;
    train_MAE_volct(i) = validationMAE;
    train_R2_volct(i) = validationR2;
    predictions_volct_i_train = validationPredictions;
    predictions_volct_i_test = abneg_age_pred_train;
    %%
    %FC ROI
%     %% Optimize and save model
%     rng(1106);
%     %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_fc17_age(abneg_train_i);
%     %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_fc300_age(abneg_train_i);
%     %[trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_fc17_age(abneg_train_i);
%     [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_roi_300_opt(abneg_train);
%     save('train_gpr_rq_roi_300_opt_model_390_220506', 'trainedModel');
%     save('train_gpr_rq_roi_300_opt_pred_390_220506', 'validationPredictions');
%     %%
    
    rng(1106);
    %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_fc17_age(abneg_train_i);
    %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_fc300_age(abneg_train_i);
    %[trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_fc17_age(abneg_train_i);
    [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_roi_300(abneg_train_i);
    % Extract ABNeg age predictions & true ages
    %abneg_age_pred = trained_gpr.predictFcn(abneg_train(:, 1:44253));
    %abneg_age_pred_train = validationPredictions;
    %abneg_age_true_train = abneg_train_i.Age;
    
    abneg_age_pred_train = trainedModel.predictFcn(abneg_test_i);%(:,1:105));%(:,1:44253));
    abneg_age_true_train = abneg_test_i.Age;
    
    % Validation R2 & MAE
    validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

    cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
    validationR2 = cormat(1,2)^2;

    train_RMSE_fc(i) = validationRMSE;
    train_MAE_fc(i) = validationMAE;
    train_R2_fc(i) = validationR2;
    predictions_fc_i_train = validationPredictions;
    predictions_fc_i_test = abneg_age_pred_train;
    
    %%
    %FC NET
    
    %% Optimize and save model
%     rng(1106);
%     %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_fc17_age(abneg_train_i);
%     %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_fc300_age(abneg_train_i);
%     [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_fc17_age_opt(abneg_train);
%     %[trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_roi_300(abneg_train_i);
%     save('train_gpr_rq_roi_17_opt_model_390_220506', 'trainedModel');
%     save('train_gpr_rq_roi_17_opt_pred_390_220506', 'validationPredictions');
%     %%
    
    rng(1106);
    %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_fc17_age(abneg_train_i);
    %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_fc300_age(abneg_train_i);
    [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_fc17_age(abneg_train_i);
    %[trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_roi_300(abneg_train_i);
     
    % Extract ABNeg age predictions & true ages
    %abneg_age_pred = trained_gpr.predictFcn(abneg_train(:, 1:44253));
    %abneg_age_pred_train = validationPredictions;
    %abneg_age_true_train = abneg_train_i.Age;
    
    abneg_age_pred_train = trainedModel.predictFcn(abneg_test_i);%(:,1:105));%(:,1:44253));
    abneg_age_true_train = abneg_test_i.Age;
    
    % Validation R2 & MAE
    validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

    cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
    validationR2 = cormat(1,2)^2;

    train_RMSE_fcnet(i) = validationRMSE;
    train_MAE_fcnet(i) = validationMAE;
    train_R2_fcnet(i) = validationR2;
    predictions_fcnet_i_train = validationPredictions;
    predictions_fcnet_i_test = abneg_age_pred_train;
    %%
%     %MERGE
%     rng(1106);
%     %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_fc17vol101_age(abneg_train_i);
%     %[trainedModel, validationRMSE, validationPredictions] = train_boost_tree_fc300vol101_age(abneg_train_i);
%     %[trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_fc17_vol101_age(abneg_train_i);
%     [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_fc300_vol101_age(abneg_train_i);
%     %[trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_volthick_111_fcharm_300(abneg_train_i);
% 
%     % Extract ABNeg age predictions & true ages
%     %abneg_age_pred = trained_gpr.predictFcn(abneg_train(:, 1:44253));
%     abneg_age_pred_train = validationPredictions;
%     abneg_age_true_train = abneg_train_i.Age;
% 
%     % Validation R2 & MAE
%     validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);
% 
%     cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
%     validationR2 = cormat(1,2)^2;
% 
%     train_RMSE_volfc(i) = validationRMSE;
%     train_MAE_volfc(i) = validationMAE;
%     train_R2_volfc(i) = validationR2;
    
    %% Stacked Vol FC tables
    stacked_table_train = table(predictions_volct_i_train, predictions_volsub_i_train, predictions_fc_i_train, predictions_fcnet_i_train);
    stacked_table_train.Age = abneg_train_i.Age;
    stacked_table_train.Properties.VariableNames = {'predictions_volct_i', 'predictions_volsub_i', 'predictions_fc_i', 'predictions_fcnet_i', 'Age'};
    
    stacked_table_test = table(predictions_volct_i_test, predictions_volsub_i_test, predictions_fc_i_test, predictions_fcnet_i_test);
    stacked_table_test.Age = abneg_test_i.Age;
    stacked_table_test.Properties.VariableNames = {'predictions_volct_i', 'predictions_volsub_i', 'predictions_fc_i', 'predictions_fcnet_i', 'Age'};

    %% Stacked gpr
    rng(1106);
    [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_stacked4_age(stacked_table_train);
    %abneg_age_pred_train = validationPredictions;
    %abneg_age_true_train = abneg_train_i.Age;
    
    abneg_age_pred_train = trainedModel.predictFcn(stacked_table_test);%(:,1:105));%(:,1:44253));
    abneg_age_true_train = stacked_table_test.Age;

    % Validation R2 & MAE
    validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

    cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
    validationR2 = cormat(1,2)^2;

    train_RMSE_volfcstack_gpr(i) = validationRMSE;
    train_MAE_volfcstack_gpr(i) = validationMAE;
    train_R2_volfcstack_gpr(i) = validationR2;
    
    %% Stacked tree
    rng(1106);
    [trainedModel, validationRMSE, validationPredictions] = train_boostedtree_stacked4_age(stacked_table_train);
    %abneg_age_pred_train = validationPredictions;
    %abneg_age_true_train = abneg_train_i.Age;
    
    abneg_age_pred_train = trainedModel.predictFcn(stacked_table_test);%(:,1:105));%(:,1:44253));
    abneg_age_true_train = stacked_table_test.Age;

    % Validation R2 & MAE
    validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

    cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
    validationR2 = cormat(1,2)^2;

    train_RMSE_volfcstack_tree(i) = validationRMSE;
    train_MAE_volfcstack_tree(i) = validationMAE;
    train_R2_volfcstack_tree(i) = validationR2;
    
    %% Stacked RF
    nSample = width(stacked_table_train) - 2;
    rng(1106);
    Mdl = TreeBagger(1000, stacked_table_train, 'Age','Method','regression', 'NumPredictorsToSample', nSample)
    abneg_age_pred_train = predict(Mdl, stacked_table_test);
    
    %[trainedModel, validationRMSE, validationPredictions] = train_boostedtree_stacked_age(stacked_table_train);
    %abneg_age_pred_train = validationPredictions;
    %abneg_age_true_train = abneg_train_i.Age;
    
    %abneg_age_pred_train = trainedModel.predictFcn(stacked_table_test);%(:,1:105));%(:,1:44253));
    abneg_age_true_train = stacked_table_test.Age;

    % Validation R2 & MAE
    validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

    cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
    validationR2 = cormat(1,2)^2;

    train_RMSE_volfcstack_rf(i) = validationRMSE;
    train_MAE_volfcstack_rf(i) = validationMAE;
    train_R2_volfcstack_rf(i) = validationR2;
    
    %% Stacked svm
    rng(1106);
    [trainedModel, validationRMSE, validationPredictions] = train_svm_cg_stacked4_age(stacked_table_train);
    %abneg_age_pred_train = validationPredictions;
    %abneg_age_true_train = abneg_train_i.Age;
    
    abneg_age_pred_train = trainedModel.predictFcn(stacked_table_test);%(:,1:105));%(:,1:44253));
    abneg_age_true_train = stacked_table_test.Age;

    % Validation R2 & MAE
    validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

    cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
    validationR2 = cormat(1,2)^2;

    train_RMSE_volfcstack_svm(i) = validationRMSE;
    train_MAE_volfcstack_svm(i) = validationMAE;
    train_R2_volfcstack_svm(i) = validationR2;
    
    %% Stacked reg
    rng(1106);
    [trainedModel, validationRMSE, validationPredictions] = train_reg_lin_stacked4_age(stacked_table_train);
    %abneg_age_pred_train = validationPredictions;
    %abneg_age_true_train = abneg_train_i.Age;
    
    abneg_age_pred_train = trainedModel.predictFcn(stacked_table_test);%(:,1:105));%(:,1:44253));
    abneg_age_true_train = stacked_table_test.Age;

    % Validation R2 & MAE
    validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

    cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
    validationR2 = cormat(1,2)^2;

    train_RMSE_volfcstack_reg(i) = validationRMSE;
    train_MAE_volfcstack_reg(i) = validationMAE;
    train_R2_volfcstack_reg(i) = validationR2;
    
    %% Stacked Vol FC tables
    stacked_table_train = table(predictions_vol_i_train, predictions_fc_i_train);
    stacked_table_train.Age = abneg_train_i.Age;
    stacked_table_train.Properties.VariableNames = {'predictions_vol_i', 'predictions_fc_i', 'Age'};
    
    stacked_table_test = table(predictions_vol_i_test, predictions_fc_i_test);
    stacked_table_test.Age = abneg_test_i.Age;
    stacked_table_test.Properties.VariableNames = {'predictions_vol_i', 'predictions_fc_i', 'Age'};
    
    %% Stacked gpr OG
    rng(1106);
    [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_stacked_age(stacked_table_train);
    %abneg_age_pred_train = validationPredictions;
    %abneg_age_true_train = abneg_train_i.Age;
    
    abneg_age_pred_train = trainedModel.predictFcn(stacked_table_test);%(:,1:105));%(:,1:44253));
    abneg_age_true_train = stacked_table_test.Age;

    % Validation R2 & MAE
    validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

    cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
    validationR2 = cormat(1,2)^2;

    train_RMSE_volfcstack2_gpr(i) = validationRMSE;
    train_MAE_volfcstack2_gpr(i) = validationMAE;
    train_R2_volfcstack2_gpr(i) = validationR2;

    %%
end

toc


fc_label = cell(nsims,1);
fc_label(:) = {'FC2-300ROI'};

fcnet_label = cell(nsims,1);
fcnet_label(:) = {'FC1-17Net'};

vol_label = cell(nsims,1);
vol_label(:) = {'Vol3-CT+Sub'};

volsub_label = cell(nsims,1);
volsub_label(:) = {'Vol2-33Sub'};

volct_label = cell(nsims,1);
volct_label(:) = {'Vol1-68CT'};

%volfc_label = cell(nsims,1);
%volfc_label(:) = {'Vol+FC Conc'};

volfcstack_gpr_label = cell(nsims,1);
volfcstack_gpr_label(:) = {'VolFC5 GPR'};

volfcstack2_gpr_label = cell(nsims,1);
volfcstack2_gpr_label(:) = {'VolFC6 GPR'};

volfcstack_tree_label = cell(nsims,1);
volfcstack_tree_label(:) = {'VolFC1 TreeBag'};

volfcstack_rf_label = cell(nsims,1);
volfcstack_rf_label(:) = {'VolFC2 TreeRF'};

volfcstack_svm_label = cell(nsims,1);
volfcstack_svm_label(:) = {'VolFC4 SVM'};

volfcstack_reg_label = cell(nsims,1);
volfcstack_reg_label(:) = {'VolFC3 Linear'};

% Setup Violin plots
MAE =   [train_MAE_fc;  train_MAE_fcnet;    train_MAE_vol;  train_MAE_volsub;  train_MAE_volct;  train_MAE_volfcstack_gpr;  train_MAE_volfcstack2_gpr;  train_MAE_volfcstack_tree;  train_MAE_volfcstack_rf;  train_MAE_volfcstack_svm;  train_MAE_volfcstack_reg];
RMSE =  [train_RMSE_fc; train_RMSE_fcnet;   train_RMSE_vol; train_RMSE_volsub; train_RMSE_volct; train_RMSE_volfcstack_gpr; train_RMSE_volfcstack2_gpr; train_RMSE_volfcstack_tree; train_RMSE_volfcstack_rf; train_RMSE_volfcstack_svm; train_RMSE_volfcstack_reg];
R2 =    [train_R2_fc;   train_R2_fcnet;     train_R2_vol;   train_R2_volsub;   train_R2_volct;   train_R2_volfcstack_gpr;   train_R2_volfcstack2_gpr;   train_R2_volfcstack_tree;   train_R2_volfcstack_rf;   train_R2_volfcstack_svm;   train_R2_volfcstack_reg];
Model = [fc_label;      fcnet_label;        vol_label;      volsub_label;      volct_label;      volfcstack_gpr_label;      volfcstack2_gpr_label;      volfcstack_tree_label;      volfcstack_rf_label;      volfcstack_svm_label;      volfcstack_reg_label];

train_table = table(Model, MAE, RMSE, R2);
%groups = {'1 Amyloid -', '2 Amyloid +', '3 CDR > 0'};

%group_label = groups(1+group);

% Violin Plot
addpath(genpath('/data/nil-bluearc/ances/millarp/matlab'));
figure;
plot_MAE = violinplot(train_table.MAE, train_table.Model); %, 'ViolinColor', [0 0 1]);
ylabel('MAE');

figure;
plot_RMSE = violinplot(train_table.RMSE, train_table.Model); %, 'ViolinColor', [0 0 1]);
ylabel('RMSE');

figure;
plot_R2 = violinplot(train_table.R2, train_table.Model); %, 'ViolinColor', [0 0 1]);
ylabel('R2');

writetable(train_table, 'comparetesting_fc300vol101stacknorm_gpr_multistack4_1000runs_220506.csv');

%% Train models
%% Train unimodal models
%% Volume
rng(1106);

[trainedModelVol, validationRMSE, validationPredictions] = train_gpr_rq_volthick_111(abneg_train);

% Extract ABNeg age predictions & true ages
%abneg_age_pred = trained_gpr.predictFcn(abneg_train(:, 1:44253));
abneg_age_pred_train = validationPredictions;
abneg_age_true_train = abneg_train.Age;

%abneg_age_pred_train = trainedModel.predictFcn(abneg_test_i);%(:,1:105));%(:,1:44253));
%abneg_age_true_train = abneg_test_i.Age;

% Validation R2 & MAE
validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
validationR2 = cormat(1,2)^2;

train_RMSE_vol = validationRMSE;
train_MAE_vol = validationMAE;
train_R2_vol = validationR2;
predictions_vol_train = validationPredictions;
%predictions_vol_i_test = abneg_age_pred_train;

%% FC ROI
rng(1106);
[trainedModelFC, validationRMSE, validationPredictions] = train_gpr_rq_roi_300(abneg_train);

% Extract ABNeg age predictions & true ages
%abneg_age_pred = trained_gpr.predictFcn(abneg_train(:, 1:44253));
abneg_age_pred_train = validationPredictions;
abneg_age_true_train = abneg_train.Age;

%abneg_age_pred_train = trainedModel.predictFcn(abneg_test);%(:,1:105));%(:,1:44253));
%abneg_age_true_train = abneg_test.Age;

% Validation R2 & MAE
validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
validationR2 = cormat(1,2)^2;

train_RMSE_fc = validationRMSE;
train_MAE_fc = validationMAE;
train_R2_fc = validationR2;
predictions_fc_train = validationPredictions;
%predictions_fc_test = abneg_age_pred_train;

%% Stacked Vol FC tables
stacked_table_train = table(predictions_vol_train, predictions_fc_train);
stacked_table_train.Age = abneg_train.Age;
stacked_table_train.Properties.VariableNames = {'predictions_vol_i', 'predictions_fc_i', 'Age'};

%% Stacked gpr
rng(1106);
[trainedModelMulti, validationRMSE, validationPredictions] = train_gpr_rq_stacked_age(stacked_table_train);
abneg_age_pred_train = validationPredictions;
abneg_age_true_train = abneg_train.Age;

%abneg_age_pred_train = trainedModel.predictFcn(stacked_table_test);%(:,1:105));%(:,1:44253));
%abneg_age_true_train = stacked_table_test.Age;

% Validation R2 & MAE
validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);

cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
validationR2 = cormat(1,2)^2;

train_RMSE_volfc = validationRMSE;
train_MAE_volfc = validationMAE;
train_R2_volfc = validationR2;

% %% Optimize Model
% rng(1106);
% Mdl = fitrauto(abnegs_norm_vol,'Age','Learners','all','OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('UseParallel',true,'Repartition',true));
% 
% %% Train model to predict age on abneg_train (access to 10-fold CV)
% 
% tic
% rng(1106);
% 
% %[trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_fc17_age(abneg_train);
% [trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_fc17_vol101_age(abneg_train);
% %[trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_roi_300(abneg_train);
% %[trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_volthick_111(abneg_train);
% %[trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_fc300_vol101_age(abneg_train);
% %[trainedModel, validationRMSE, validationPredictions] = train_gpr_rq_volthick_111_fcharm_300(abneg_train);
% 
% toc
% % about 3 minutes to train
% 
% % Extract ABNeg age predictions & true ages
% %abneg_age_pred = trained_gpr.predictFcn(abneg_train(:, 1:44253));
% abneg_age_pred_train = validationPredictions;
% abneg_age_true_train = abneg_train.Age;
% 
% % Validation R2 & MAE
% validationMAE = mae(abneg_age_pred_train - abneg_age_true_train);
% 
% cormat = corrcoef(abneg_age_pred_train, abneg_age_true_train);
% validationR2 = cormat(1,2)^2;

%% Compare true & predicted ages
%% FC
figure
scatter(abneg_age_true_train, predictions_fc_train, 'b.')
grid on;

hold on
%xlim([min([abneg_age_true_train; abneg_age_pred_train]), max([abneg_age_true_train; abneg_age_pred_train])])
%ylim([min([abneg_age_true_train; abneg_age_pred_train]), max([abneg_age_true_train; abneg_age_pred_train])])
xlim([10,90])
ylim([10,90])
plot(xlim, ylim, '--k')

%ref = refline(1,0);
%set(ref, 'color', 'k');

lines = lsline;
set(lines(1), 'color', 'b');

title('FC-Predicted Brain Age');
xlabel('True Age');
ylabel('Predicted Age');

%% Vol
figure
scatter(abneg_age_true_train, predictions_vol_train, 'b.')
grid on;

hold on
%xlim([min([abneg_age_true_train; abneg_age_pred_train]), max([abneg_age_true_train; abneg_age_pred_train])])
%ylim([min([abneg_age_true_train; abneg_age_pred_train]), max([abneg_age_true_train; abneg_age_pred_train])])
xlim([10,90])
ylim([10,90])
plot(xlim, ylim, '--k')

%ref = refline(1,0);
%set(ref, 'color', 'k');

lines = lsline;
set(lines(1), 'color', 'b');

title('Vol-Predicted Brain Age');
xlabel('True Age');
ylabel('Predicted Age');

%% Merged Vol+FC
figure
scatter(abneg_age_true_train, abneg_age_pred_train, 'b.')
grid on;

hold on
%xlim([min([abneg_age_true_train; abneg_age_pred_train]), max([abneg_age_true_train; abneg_age_pred_train])])
%ylim([min([abneg_age_true_train; abneg_age_pred_train]), max([abneg_age_true_train; abneg_age_pred_train])])
xlim([10,90])
ylim([10,90])
plot(xlim, ylim, '--k')

%ref = refline(1,0);
%set(ref, 'color', 'k');

lines = lsline;
set(lines(1), 'color', 'b');

title('Vol+FC-Predicted Brain Age');
xlabel('True Age');
ylabel('Predicted Age');


%% Test on ABNeg data
predictions_fc_test = trainedModelFC.predictFcn(abneg_test);
predictions_vol_test = trainedModelVol.predictFcn(abneg_test);
abneg_age_true = abneg_test.Age;

stacked_table_test = table(predictions_vol_test, predictions_fc_test);
stacked_table_test.Age = abneg_test.Age;
stacked_table_test.Properties.VariableNames = {'predictions_vol_i', 'predictions_fc_i', 'Age'};

abneg_age_pred = trainedModelMulti.predictFcn(stacked_table_test);

%% Compare true & predicted ages - ABNeg Test Set

figure;
plot_abneg = scatter(abneg_age_true, abneg_age_pred, 'b.');
hold on;

grid on;
plot(xlim, ylim, '--k')

%ref = refline(1,0);
%set(ref, 'color', 'k');
%set(ref, 'LineStyle', ':');

lines = lsline;
set(lines(1), 'color', 'b');
%set(lines(2), 'color', 'g');
%set(lines(1), 'color', 'r');
%set(lines(2), 'LineStyle', '--');
%set(lines(1), 'LineStyle', '--');

title('Vol+FC-Predicted Brain Age');
xlabel('True Age');
ylabel('Predicted Age');
%legend({'Amyloid -', 'Amyloid +', 'CDR > 0'}, 'Location', 'northwest');

testMAE_abneg = mae(abneg_age_pred - abneg_age_true);
testRMSE_abneg = sqrt(mean((abneg_age_pred - abneg_age_true).^2));

cormat = corrcoef(abneg_age_pred,  abneg_age_true);
testR2_abneg = cormat(1,2)^2;

%% Test on ABPos data
predictions_fc_abpos = trainedModelFC.predictFcn(abposs);
predictions_vol_abpos = trainedModelVol.predictFcn(abposs);
abpos_age_true = abposs.Age;

stacked_table_abpos = table(predictions_vol_abpos, predictions_fc_abpos);
stacked_table_abpos.Age = abposs.Age;
stacked_table_abpos.Properties.VariableNames = {'predictions_vol_i', 'predictions_fc_i', 'Age'};

abpos_age_pred = trainedModelMulti.predictFcn(stacked_table_abpos);



%% Test on CDR >0 data
predictions_fc_cdr = trainedModelFC.predictFcn(cdr051s);
predictions_vol_cdr = trainedModelVol.predictFcn(cdr051s);
cdr051_age_true = cdr051s.Age;

stacked_table_cdr = table(predictions_vol_cdr, predictions_fc_cdr);
stacked_table_cdr.Age = cdr051s.Age;
stacked_table_cdr.Properties.VariableNames = {'predictions_vol_i', 'predictions_fc_i', 'Age'};

cdr051_age_pred = trainedModelMulti.predictFcn(stacked_table_cdr);

%% Compare true & predicted ages - all groups

figure;
plot_abneg = scatter(abneg_age_true, abneg_age_pred, 'bo');
hold on;

plot_abpos = scatter(abpos_age_true, abpos_age_pred, 'go');

plot_cdr051 = scatter(cdr051_age_true, cdr051_age_pred, 'ro');

grid on;

ref = refline(1,0);
set(ref, 'color', 'k');
%set(ref, 'LineStyle', ':');

lines = lsline;
set(lines(3), 'color', 'b');
set(lines(2), 'color', 'g');
set(lines(1), 'color', 'r');
%set(lines(2), 'LineStyle', '--');
%set(lines(1), 'LineStyle', '--');

title('Vol+FC-Predicted Brain Age');
xlabel('True Age');
ylabel('Predicted Age');
legend({'Amyloid -', 'Amyloid +', 'CDR > 0'}, 'Location', 'northwest');

%% Compare true & predicted ages - ab neg vs ab pos

figure;
plot_abneg = scatter(abneg_age_true, abneg_age_pred, 'bo');
hold on;

plot_abpos = scatter(abpos_age_true, abpos_age_pred, 'go');

%plot_cdr051 = scatter(cdr051_age_true, cdr051_age_pred, 'ro');

grid on;

ref = refline(1,0);
set(ref, 'color', 'k');
%set(ref, 'LineStyle', ':');

lines = lsline;
set(lines(2), 'color', 'b');
set(lines(1), 'color', 'g');
%set(lines(1), 'color', 'r');
%set(lines(2), 'LineStyle', '--');
%set(lines(1), 'LineStyle', '--');

title('Vol+FC-Predicted Brain Age');
xlabel('True Age');
ylabel('Predicted Age');
legend({'Amyloid -', 'Amyloid +'}, 'Location', 'northwest');

%% Compare true & predicted ages - CDR 0 vs >0

figure;
plot_abneg = scatter([abneg_age_true; abpos_age_true], [abneg_age_pred; abpos_age_pred], 'bo');
hold on;

%plot_abpos = scatter(abpos_age_true, abpos_age_pred, 'go');

plot_cdr051 = scatter(cdr051_age_true, cdr051_age_pred, 'ro');

grid on;

ref = refline(1,0);
set(ref, 'color', 'k');
%set(ref, 'LineStyle', ':');

lines = lsline;
set(lines(2), 'color', 'b');
%set(lines(2), 'color', 'g');
set(lines(1), 'color', 'r');
%set(lines(2), 'LineStyle', '--');
%set(lines(1), 'LineStyle', '--');

title('FC-Predicted Brain Age');
xlabel('True Age');
ylabel('Predicted Age');
legend({'CDR = 0', 'CDR > 0'}, 'Location', 'northwest');

%% Combine ABNeg and ABPos and CDR > 0s
abneg_abs = repelem(0, length(abneg_age_true)).';
abpos_abs = repelem(1, length(abpos_age_true)).';
cdr051_abs = repelem(2, length(cdr051_age_true)).';

group = [abneg_abs; abpos_abs; cdr051_abs];
age = [abneg_age_true; abpos_age_true; cdr051_age_true];
age_pred = [abneg_age_pred; abpos_age_pred; cdr051_age_pred];

ancova_table = table(group, age, age_pred);


%%
% abneg_error = abneg_age_pred - abneg_age_true;
% abpos_error = abpos_age_pred - abpos_age_true;
% abneg_table = ancova_table(ancova_table.ab == 0, :);
% abneg_model = fitlm(abneg_table, 'age_pred ~ age');
% abneg_error = abneg_model.Residuals.Raw;
% 
% abpos_table = ancova_table(ancova_table.ab == 1, :);
% abpos_model = fitlm(abpos_table, 'age_pred ~ age');
% abpos_error = abpos_model.Residuals.Raw;

group_model = fitlm(ancova_table, 'age_pred ~ age');
abneg_error = group_model.Residuals.Raw(ancova_table.group == 0);
abpos_error = group_model.Residuals.Raw(ancova_table.group == 1);
cdr051_error = group_model.Residuals.Raw(ancova_table.group == 2);

%%
figure;
%nbins = 50;
hist_abneg = histogram(abneg_error, 35, 'FaceColor', 'b');
hold on;
hist_abpos = histogram(abpos_error, 35, 'FaceColor', 'g');
hist_cdr051 = histogram(cdr051_error, 35, 'FaceColor', 'r');



%% Compare resid predicted age to age gap and resid age gap

ancova_table.agegap = ancova_table.age_pred - ancova_table.age;

agegap_model = fitlm(ancova_table, 'agegap ~ age');
ancova_table.agegapreg = agegap_model.Residuals.Raw;

%cdr0_agegapreg = agegap_model.Residuals.Raw(ancova_table.cdr == 0);

abneg_agegap = ancova_table.agegap(ancova_table.group == 0);
abpos_agegap = ancova_table.agegap(ancova_table.group == 1);
cdr051_agegap = ancova_table.agegap(ancova_table.group == 2);

abneg_agegapreg = ancova_table.agegapreg(ancova_table.group == 0);
abpos_agegapreg = ancova_table.agegapreg(ancova_table.group == 1);
cdr051_agegapreg = ancova_table.agegapreg(ancova_table.group == 2);


%% Compare AGE GAP ESTIMATION as a function of true age
%close all;

figure;
plotneg = scatter(abneg_age_true, abneg_agegap, 'bo');
hold on;
plotpos = scatter(abpos_age_true, abpos_agegap, 'go');

plot051 = scatter(cdr051_age_true, cdr051_agegap, 'ro');
grid on;

ref = refline(0,0);
set(ref, 'color', 'k');
%set(ref, 'LineStyle', ':');

lines = lsline;
set(lines(3), 'color', 'b');
set(lines(2), 'color', 'g');
set(lines(1), 'color', 'r');
%set(lines(2), 'LineStyle', '--');
%set(lines(1), 'LineStyle', '--');

title('FC-Predicted Brain Age');
xlabel('True Age');
ylabel('Predicted - True Age');
legend({'Amyloid -', 'Amyloid +', 'CDR > 0'}, 'Location', 'northeast');

%% Compare RESIDUAL AGE GAP as a function of true age
%close all;

figure;
plotneg = scatter(abneg_age_true, abneg_agegapreg, 'bo');
hold on;
plotpos = scatter(abpos_age_true, abpos_agegapreg, 'go');

plot051 = scatter(cdr051_age_true, cdr051_agegapreg, 'ro');
grid on;

ref = refline(0,0);
set(ref, 'color', 'k');
%set(ref, 'LineStyle', ':');

lines = lsline;
set(lines(3), 'color', 'b');
set(lines(2), 'color', 'g');
set(lines(1), 'color', 'r');
%set(lines(2), 'LineStyle', '--');
%set(lines(1), 'LineStyle', '--');

title('FC-Predicted Brain Age');
xlabel('True Age');
ylabel('Residual Age Gap');
legend({'Amyloid -', 'Amyloid +', 'CDR > 0'}, 'Location', 'northeast');

%% Compare RESIDUAL PREDICTED AGE as a function of true age
%close all;

figure;
plotneg = scatter(abneg_age_true, abneg_error, 'bo');
hold on;
plotpos = scatter(abpos_age_true, abpos_error, 'go');

plot051 = scatter(cdr051_age_true, cdr051_error, 'ro');
grid on;

ref = refline(0,0);
set(ref, 'color', 'k');
%set(ref, 'LineStyle', ':');

lines = lsline;
set(lines(3), 'color', 'b');
set(lines(2), 'color', 'g');
set(lines(1), 'color', 'r');
%set(lines(2), 'LineStyle', '--');
%set(lines(1), 'LineStyle', '--');

title('FC-Predicted Brain Age');
xlabel('True Age');
ylabel('Residual Predicted Age');
legend({'Amyloid -', 'Amyloid +', 'CDR > 0'}, 'Location', 'northeast');


%% Compare AGE GAP ESTIMATION as a function of RESIDUAL PREDICTED AGE
%close all;

figure;
plotneg = scatter(abneg_error, abneg_agegap, 'bo');
hold on;
plotpos = scatter(abpos_error, abpos_agegap, 'go');

plot051 = scatter(cdr051_error, cdr051_agegap, 'ro');
grid on;

ref = refline(0,0);
set(ref, 'color', 'k');
%set(ref, 'LineStyle', ':');

lines = lsline;
set(lines(3), 'color', 'b');
set(lines(2), 'color', 'g');
set(lines(1), 'color', 'r');
%set(lines(2), 'LineStyle', '--');
%set(lines(1), 'LineStyle', '--');

title('FC-Predicted Brain Age');
xlabel('Residual Predicted Age');
ylabel('Predicted Age - True Age');
legend({'Amyloid -', 'Amyloid +', 'CDR > 0'}, 'Location', 'northwest');

%% Export Ancova table (true age, predicted age, age gap, residual age gap, residual predicted age)
ancova_table.SessionID = [abneg_test_labels.SessionID; abposs_labels.SessionID; cdr051s_labels.SessionID];
ancova_table.age_pred_fc = [predictions_fc_test; predictions_fc_abpos; predictions_fc_cdr];
ancova_table.age_pred_vol = [predictions_vol_test; predictions_vol_abpos; predictions_vol_cdr];

ancova_table.agegap_fc = ancova_table.age_pred_fc - ancova_table.age;
agegap_fc_model = fitlm(ancova_table, 'agegap_fc ~ age');
ancova_table.agegapreg_fc = agegap_fc_model.Residuals.Raw;

ancova_table.agegap_vol = ancova_table.age_pred_vol - ancova_table.age;
agegap_vol_model = fitlm(ancova_table, 'agegap_vol ~ age');
ancova_table.agegapreg_vol = agegap_vol_model.Residuals.Raw;

%%

writetable(ancova_table, 'volfcpredictedages_abNvabPvsym_fc300vol101harm_153v143v182_210615.csv');

