%% ============================================================
% Fe-Cr-Ni-Mn 四元合金：BP 神经网络 + GA 逆向设计
% 方法：BP神经网络 + 多特征输入比较 + 约束遗传算法(GA)
%
% 改进方案：
%   1) 机器学习阶段只预测剪切模量 G
%   2) SFE 不进入机器学习目标函数，仅用于后续 MD 复算与机理分析
%   3) 将已完成 MD 复算的候选点作为反馈样本加入训练集
%   4) 使用 trainbr 正则化 BP 网络，降低小样本过拟合风险
%   5) 采用 BP 集成模型：输出预测均值与模型分歧
%   6) GA 适应度 = G预测均值 - 模型分歧惩罚 - 适用域距离惩罚
%   7) 可选训练 SFE 辅助集成模型，仅用于 TopK 自动分散选点，不作为最终判据
%   8) 从高 G 候选池中自动输出兼顾成分/SFE分散性的 Top10，后续全部 MD 复算 G / SFE / 拉伸验证
%
% 约束：
%   1. 5% <= 每个元素 <= 65%
%   2. Fe + Cr + Ni + Mn = 100%
%   3. VEC >= 7.8
%% ============================================================

clear; clc; close all;
rng(42, 'twister');

%% ===================== 0. 用户参数设置 ======================
% ---------- 选文件 ----------
[file, path] = uigetfile({'*.xlsx;*.xls', 'Excel Files (*.xlsx, *.xls)'}, ...
                         '请选择 Fe-Cr-Ni-Mn 数据文件');
if isequal(file,0)
    error('未选择数据文件，程序终止。');
end
filename = fullfile(path, file);

% 重复训练次数（每个特征组重复训练多次，减少随机划分影响）
N_REPEATS = 10;

% 训练集比例
train_ratio = 0.85;

% BP 网络参数
% 使用更小的网络 + Bayesian regularization，降低小样本外推虚高风险
hiddenLayerSize = [8 4];
trainFcn = 'trainbr';          % Bayesian regularization，较 trainlm 更稳健

% 最终集成模型数量
ENSEMBLE_N = 8;

% GA 参数
POP_SIZE   = 120;
GENS       = 150;
MUT_RATE   = 0.12;
CROSS_RATE = 0.85;
RANGE      = [0.05, 0.65];     % 每个元素原子分数范围

% 候选池输出参数
TOPK       = 1;                % 最终输出前 TOPK 个候选组分
POOLK      = 1000;             % 先保留较大的高 G 候选池，再做自动分散选点
MIN_DIST   = 0.10;             % 最终 TopK 的成分空间最小距离，避免组分过于相似
G_WINDOW   = 8.0;              % 高 G 候选池窗口：允许从略低于最高预测 G 的区域中选分散点

% 自动分散选点权重：不是人工挑选，而是固定算法从高 G 候选池输出 TopK
% 本版优先使用“高 G + 成分空间分散”，不强依赖 SFE 预测，因为 SFE 仍以 MD 复算为准
COMP_DIVERSITY_WEIGHT = 1.20;  % 成分空间分散权重
SFE_DIVERSITY_WEIGHT  = 0.00;  % SFE 预测不参与最终选点，避免辅助模型误导
USE_SFE_AUX_DIVERSITY = false; % SFE 不作为机器学习筛选项；最终由 MD 复算

% 是否加入已完成 MD 复算的反馈样本
% 这些点用于纠正 BP 在 GA 新成分区域的外推误差
USE_FEEDBACK_DATA = false;

% 适用域与集成不确定性惩罚参数
% 注意：这里不再惩罚“G 高于训练集最大值”，而是惩罚模型分歧大、远离训练数据的候选点
LAMBDA_UNC  = 0.50;            % BP 集成标准差惩罚权重
LAMBDA_DIST = 10.0;            % 特征空间适用域距离惩罚权重
DIST_TH     = 0.45;            % 归一化特征空间距离阈值

% 是否显示训练窗口
showTrainWindow = false;

% 是否在存在 SFE 列时过滤掉 SFE<=0 的样本（可选）
USE_SFE_FILTER_IF_AVAILABLE = true;

%% ===================== 1. 读取与预处理数据 ======================
opts = detectImportOptions(filename);
data_table = readtable(filename, opts);

% -------- 检查必要列名 --------
required_cols = {'Fe','Cr','Ni','Mn','G_GPa'};
for k = 1:numel(required_cols)
    if ~ismember(required_cols{k}, data_table.Properties.VariableNames)
        error('数据表缺少必要列: %s', required_cols{k});
    end
end

raw_comps = [data_table.Fe, data_table.Cr, data_table.Ni, data_table.Mn];
raw_G     = data_table.G_GPa;

hasSFE = ismember('SFE_mJm2', data_table.Properties.VariableNames);
if hasSFE
    raw_SFE = data_table.SFE_mJm2;
else
    raw_SFE = nan(size(raw_G));
end

% 若输入为百分数（总和接近100），自动转成原子分数
row_sums = sum(raw_comps, 2);
if mean(row_sums, 'omitnan') > 1.5
    raw_comps = raw_comps / 100;
end

% 剔除基本非法数据
valid_idx = all(~isnan(raw_comps), 2) & ~isnan(raw_G);
raw_comps = raw_comps(valid_idx, :);
raw_G     = raw_G(valid_idx);
raw_SFE   = raw_SFE(valid_idx);

% 若存在 SFE 列，可选：剔除 SFE<=0 的不稳定样本
if hasSFE && USE_SFE_FILTER_IF_AVAILABLE
    idx2 = raw_SFE > 0;
    comps = raw_comps(idx2, :);
    G     = raw_G(idx2);
    SFE   = raw_SFE(idx2);
    fprintf('检测到 SFE 列，已按 SFE>0 进行过滤。\n');
else
    comps = raw_comps;
    G     = raw_G;
    SFE   = raw_SFE;
    if hasSFE
        fprintf('检测到 SFE 列，但当前未使用 SFE>0 过滤。\n');
    else
        fprintf('未检测到 SFE 列，本次仅使用 G 数据建模。\n');
    end
end

fprintf('用于训练的数据点数: %d\n', length(G));

% 再次归一化组分，防止和不严格等于 1
for i = 1:size(comps,1)
    comps(i,:) = comps(i,:) / sum(comps(i,:));
end

%% ===================== 1.1 加入 MD 反馈样本（可选） ======================
% Public code release: no feedback dataset is embedded in this script.
% If feedback samples are to be used, select an external Excel/CSV file
% containing the columns: Fe, Cr, Ni, Mn, G_GPa.
% Compositions may be fractions (sum ~1) or at.% (sum ~100).
if USE_FEEDBACK_DATA
    [fb_file, fb_path] = uigetfile({'*.xlsx;*.xls;*.csv', ...
        'Data Files (*.xlsx, *.xls, *.csv)'}, ...
        '请选择可选的 MD 反馈数据文件');
    if isequal(fb_file,0)
        warning('未选择反馈数据文件，继续使用原始训练集。');
    else
        fb_name = fullfile(fb_path, fb_file);
        fb_table = readtable(fb_name);
        fb_required = {'Fe','Cr','Ni','Mn','G_GPa'};
        for k = 1:numel(fb_required)
            if ~ismember(fb_required{k}, fb_table.Properties.VariableNames)
                error('反馈数据缺少必要列: %s', fb_required{k});
            end
        end

        fb_comps = [fb_table.Fe, fb_table.Cr, fb_table.Ni, fb_table.Mn];
        fb_G = fb_table.G_GPa;
        fb_sum = sum(fb_comps,2);
        if mean(fb_sum, 'omitnan') > 1.5
            fb_comps = fb_comps / 100;
        end
        valid_fb = all(isfinite(fb_comps),2) & isfinite(fb_G);
        fb_comps = fb_comps(valid_fb,:);
        fb_G = fb_G(valid_fb);
        for i = 1:size(fb_comps,1)
            fb_comps(i,:) = fb_comps(i,:) / sum(fb_comps(i,:));
        end

        comps = [comps; fb_comps];
        G = [G; fb_G];
        if exist('SFE','var')
            SFE = [SFE; nan(size(fb_G))];
        end
        fprintf('已加入 MD 反馈样本数: %d\n', size(fb_comps,1));
    end
end

% 用于 OOD 惩罚的训练集 G 范围
G_MIN_TRAIN = min(G);
G_MAX_TRAIN = max(G);
fprintf('训练集 G 范围: %.4f ~ %.4f GPa\n\n', G_MIN_TRAIN, G_MAX_TRAIN);

%% ===================== 2. 基础物理参数定义 ======================
% 元素顺序固定: [Fe, Cr, Ni, Mn]
params.elem_names = {'Fe','Cr','Ni','Mn'};

% 原子半径 (pm)
params.r = [124, 128, 124, 127];

% 熔点 Tm (K)
params.Tm = [1811, 2180, 1728, 1519];

% 价电子浓度 VEC
params.VEC = [8, 6, 10, 7];

% 二元混合焓矩阵 H_ij (kJ/mol)
% 顺序：[Fe, Cr, Ni, Mn]
H = zeros(4,4);
H(1,2) = -1;   H(2,1) = H(1,2);   % Fe-Cr
H(1,3) = -2;   H(3,1) = H(1,3);   % Fe-Ni
H(1,4) =  0;   H(4,1) = H(1,4);   % Fe-Mn
H(2,3) = -7;   H(3,2) = H(2,3);   % Cr-Ni
H(2,4) =  2;   H(4,2) = H(2,4);   % Cr-Mn
H(3,4) = -8;   H(4,3) = H(3,4);   % Ni-Mn
params.HmixPair = H;

% 气体常数
params.R = 8.314;

%% ===================== 3. 构造候选特征组 ======================
featureModes = {
    'comp3', ...        % [Fe Cr Ni]
    'comp4', ...        % [Fe Cr Ni Mn]
    'phys_only', ...    % [delta Hmix Smix Omega VEC Tm_bar]
    'comp3_phys', ...   % [Fe Cr Ni + phys]
    'comp4_phys' ...    % [Fe Cr Ni Mn + phys]
    };

nModes = numel(featureModes);

results = struct();
summary_cell = cell(nModes, 8);

fprintf('================= 开始不同特征组模型比较（只预测 G） =================\n');

for m = 1:nModes
    modeName = featureModes{m};
    fprintf('\n>>> 特征组: %s\n', modeName);

    X_all = buildFeatures(comps, modeName, params);
    Y_all = G(:);

    R2_train_list  = zeros(N_REPEATS,1);
    R2_test_list   = zeros(N_REPEATS,1);
    RMSE_test_list = zeros(N_REPEATS,1);
    MAE_test_list  = zeros(N_REPEATS,1);

    best_this_mode.R2_test = -inf;
    best_this_mode.net = [];
    best_this_mode.input_ps = [];
    best_this_mode.output_ps = [];
    best_this_mode.modeName = modeName;
    best_this_mode.X_all = X_all;
    best_this_mode.Y_all = Y_all;

    for rep = 1:N_REPEATS
        n = size(X_all,1);
        idx = randperm(n);
        train_end = max(2, round(train_ratio * n));

        train_idx = idx(1:train_end);
        test_idx  = idx(train_end+1:end);

        if isempty(test_idx)
            test_idx  = train_idx(end);
            train_idx = train_idx(1:end-1);
        end

        X_train = X_all(train_idx, :);
        Y_train = Y_all(train_idx);
        X_test  = X_all(test_idx, :);
        Y_test  = Y_all(test_idx);

        % 仅用训练集做归一化
        [X_train_n, input_ps] = mapminmax(X_train', 0, 1);
        X_train_n = X_train_n';
        X_test_n  = mapminmax('apply', X_test', input_ps);
        X_test_n  = X_test_n';

        [Y_train_n, output_ps] = mapminmax(Y_train', 0, 1);
        Y_train_n = Y_train_n';

        % 建立 BP 网络
        net = fitnet(hiddenLayerSize, trainFcn);
        net.divideFcn = 'dividetrain';
        net.trainParam.epochs = 2500;
        net.trainParam.goal = 1e-6;
        net.trainParam.min_grad = 1e-7;
        net.trainParam.showWindow = showTrainWindow;
        net.performFcn = 'mse';

        % 训练
        net = train(net, X_train_n', Y_train_n');

        % 预测
        pred_train_n = net(X_train_n');
        pred_test_n  = net(X_test_n');

        Y_pred_train = mapminmax('reverse', pred_train_n, output_ps)';
        Y_pred_test  = mapminmax('reverse', pred_test_n,  output_ps)';

        % 评价
        R2_train  = calcR2(Y_train, Y_pred_train);
        R2_test   = calcR2(Y_test,  Y_pred_test);
        RMSE_test = sqrt(mean((Y_pred_test - Y_test).^2));
        MAE_test  = mean(abs(Y_pred_test - Y_test));

        R2_train_list(rep)  = R2_train;
        R2_test_list(rep)   = R2_test;
        RMSE_test_list(rep) = RMSE_test;
        MAE_test_list(rep)  = MAE_test;

        % 保留当前特征组中最好的单个模型
        if R2_test > best_this_mode.R2_test
            best_this_mode.R2_test = R2_test;
            best_this_mode.net = net;
            best_this_mode.input_ps = input_ps;
            best_this_mode.output_ps = output_ps;
            best_this_mode.train_idx = train_idx;
            best_this_mode.test_idx  = test_idx;
            best_this_mode.Y_test = Y_test;
            best_this_mode.Y_pred_test = Y_pred_test;
            best_this_mode.Y_train = Y_train;
            best_this_mode.Y_pred_train = Y_pred_train;
        end
    end

    % 汇总
    results(m).mode = modeName;
    results(m).R2_train_mean  = mean(R2_train_list);
    results(m).R2_train_std   = std(R2_train_list);
    results(m).R2_test_mean   = mean(R2_test_list);
    results(m).R2_test_std    = std(R2_test_list);
    results(m).RMSE_test_mean = mean(RMSE_test_list);
    results(m).RMSE_test_std  = std(RMSE_test_list);
    results(m).MAE_test_mean  = mean(MAE_test_list);
    results(m).MAE_test_std   = std(MAE_test_list);
    results(m).bestModel      = best_this_mode;

    summary_cell{m,1} = modeName;
    summary_cell{m,2} = mean(R2_train_list);
    summary_cell{m,3} = std(R2_train_list);
    summary_cell{m,4} = mean(R2_test_list);
    summary_cell{m,5} = std(R2_test_list);
    summary_cell{m,6} = mean(RMSE_test_list);
    summary_cell{m,7} = mean(MAE_test_list);
    summary_cell{m,8} = best_this_mode.R2_test;

    fprintf('平均训练集 R^2 = %.4f ± %.4f\n', mean(R2_train_list), std(R2_train_list));
    fprintf('平均测试集 R^2 = %.4f ± %.4f\n', mean(R2_test_list),  std(R2_test_list));
    fprintf('平均测试集 RMSE = %.4f\n', mean(RMSE_test_list));
    fprintf('平均测试集 MAE  = %.4f\n', mean(MAE_test_list));
    fprintf('该组最佳单次测试 R^2 = %.4f\n', best_this_mode.R2_test);
end

%% ===================== 4. 输出模型比较结果 ======================
summary_table = cell2table(summary_cell, ...
    'VariableNames', {'FeatureMode','R2TrainMean','R2TrainStd', ...
                      'R2TestMean','R2TestStd','RMSETestMean', ...
                      'MAETestMean','BestSingleR2Test'});

disp(' ');
disp('================= 不同特征组模型表现汇总 =================');
disp(summary_table);

% 按平均测试集 R2 选择最优特征组（以 G 模型为准）
all_R2_mean = arrayfun(@(s) s.R2_test_mean, results);
[~, best_mode_idx] = max(all_R2_mean);
best_mode_name = results(best_mode_idx).mode;

fprintf('\n====================================================\n');
fprintf('自动选择的最佳特征组: %s\n', best_mode_name);
fprintf('其平均测试集 R^2 = %.4f ± %.4f\n', ...
    results(best_mode_idx).R2_test_mean, results(best_mode_idx).R2_test_std);
fprintf('====================================================\n');

%% ===================== 5. 可视化：不同特征组性能对比 ======================
figure('Name','不同特征组模型比较','Color','w');
bar(all_R2_mean, 'FaceColor', [0.3 0.6 0.85]); hold on;
errorbar(1:nModes, all_R2_mean, arrayfun(@(s) s.R2_test_std, results), ...
    'k.', 'LineWidth', 1.5);
set(gca, 'XTick', 1:nModes, 'XTickLabel', featureModes, 'FontSize', 11);
ylabel('Mean Test R^2');
title('Comparison of Feature Sets for BP Prediction of G');
grid on;

%% ===================== 6. 绘制最佳特征组最优单次散点图 ======================
best_single = results(best_mode_idx).bestModel;

figure('Name','最佳模型预测对比','Color','w');
scatter(best_single.Y_train, best_single.Y_pred_train, 55, 'b', 'filled'); hold on;
scatter(best_single.Y_test,  best_single.Y_pred_test,  65, 'r', 'filled');

gmin = min([best_single.Y_train; best_single.Y_test; ...
            best_single.Y_pred_train; best_single.Y_pred_test]);
gmax = max([best_single.Y_train; best_single.Y_test; ...
            best_single.Y_pred_train; best_single.Y_pred_test]);

plot([gmin gmax], [gmin gmax], 'k--', 'LineWidth', 1.5);
xlabel('Simulated G (GPa)');
ylabel('Predicted G (GPa)');
title(sprintf('Best Feature Set: %s | Best Test R^2 = %.4f', ...
    best_mode_name, best_single.R2_test));
legend('Train', 'Test', 'y = x', 'Location', 'best');
grid on; box on;

%% ===================== 6.1 导出最佳单次模型预测对比数据（供 Origin 作图） ======================
% 该表用于重画“Predicted G vs Simulated G”图：
%   Set = Train/Test，用于区分蓝色训练集和红色测试集；
%   Simulated_G = MD 计算/模拟的 G；
%   Predicted_G = BP 模型预测的 G；
%   Ref_X, Ref_Y = y = x 参考线的两个端点。
nTrain = numel(best_single.Y_train);
nTest  = numel(best_single.Y_test);

Set = [repmat({'Train'}, nTrain, 1); repmat({'Test'}, nTest, 1)];
PointType = [ones(nTrain, 1); 2 * ones(nTest, 1)];

Simulated_G = [best_single.Y_train(:); best_single.Y_test(:)];
Predicted_G = [best_single.Y_pred_train(:); best_single.Y_pred_test(:)];
Residual_G = Simulated_G - Predicted_G;

Ref_X = nan(numel(Simulated_G), 1);
Ref_Y = nan(numel(Simulated_G), 1);
gmin_ref = floor(min([Simulated_G; Predicted_G]) / 5) * 5;
gmax_ref = ceil(max([Simulated_G; Predicted_G]) / 5) * 5;
Ref_X(1:2) = [gmin_ref; gmax_ref];
Ref_Y(1:2) = [gmin_ref; gmax_ref];

BPGA_prediction_origin = table(Set, PointType, Simulated_G, Predicted_G, Residual_G, Ref_X, Ref_Y);
writetable(BPGA_prediction_origin, 'BP_GA_best_single_prediction_for_Origin.csv');

fprintf('\n最佳单次 BP-GA 预测对比数据已保存为: BP_GA_best_single_prediction_for_Origin.csv\n');

% 同时保存该图，便于和 Origin 重画结果对比
saveas(gcf, 'BP_GA_best_single_prediction_scatter.png');


%% ===================== 7. 用最佳特征组在全数据上训练 BP 集成 G 模型 ======================
fprintf('\n================= 在全数据上训练最终 BP 集成 G 模型 =================\n');

X_best_all = buildFeatures(comps, best_mode_name, params);
Y_best_all = G(:);

[X_best_n, final_input_ps] = mapminmax(X_best_all', 0, 1);
X_best_n = X_best_n';
[Y_best_n, final_output_ps] = mapminmax(Y_best_all', 0, 1);
Y_best_n = Y_best_n';

final_nets = cell(ENSEMBLE_N, 1);

for e = 1:ENSEMBLE_N
    rng(1000 + e, 'twister');

    net = fitnet(hiddenLayerSize, trainFcn);
    net.divideFcn = 'dividetrain';
    net.trainParam.epochs = 2500;
    net.trainParam.goal = 1e-6;
    net.trainParam.min_grad = 1e-7;
    net.trainParam.showWindow = showTrainWindow;
    net.performFcn = 'mse';

    final_nets{e} = train(net, X_best_n', Y_best_n');
end

% 集成模型对全数据的拟合
[Y_fit, Y_fit_std] = predictGEnsemble(final_nets, X_best_n, final_output_ps);

R2_full = calcR2(Y_best_all, Y_fit);
RMSE_full = sqrt(mean((Y_fit - Y_best_all).^2));
MAE_full  = mean(abs(Y_fit - Y_best_all));

fprintf('最终 BP 集成 G 模型（全数据）R^2 = %.4f\n', R2_full);
fprintf('最终 BP 集成 G 模型（全数据）RMSE = %.4f\n', RMSE_full);
fprintf('最终 BP 集成 G 模型（全数据）MAE = %.4f\n', MAE_full);
fprintf('全数据平均模型分歧 std = %.4f GPa\n', mean(Y_fit_std));

% 保存模型
save('final_G_ensemble_model.mat', 'final_nets', 'final_input_ps', 'final_output_ps', ...
     'best_mode_name', 'hiddenLayerSize', 'trainFcn', 'ENSEMBLE_N', ...
     'X_best_n', 'Y_best_all', 'params');

%% ===================== 7.1 训练 SFE 辅助集成模型（仅用于自动分散选点） ======================
% 注意：SFE 辅助模型不参与最终性能判定，也不作为硬约束。
% 它只用于避免 TopK 全部集中在相近 SFE 水平；最终 SFE 仍以 MD 复算为准。
hasSFE_aux_model = false;
sfe_nets = {};
sfe_input_ps = [];
sfe_output_ps = [];

if USE_SFE_AUX_DIVERSITY && exist('SFE','var')
    idx_sfe = isfinite(SFE) & (SFE > 0);
    if sum(idx_sfe) >= 20
        fprintf('\n================= 训练 SFE 辅助 BP 集成模型（仅用于分散选点） =================\n');
        X_sfe_all = buildFeatures(comps(idx_sfe,:), best_mode_name, params);
        Y_sfe_all = SFE(idx_sfe);

        [X_sfe_n, sfe_input_ps] = mapminmax(X_sfe_all', 0, 1);
        X_sfe_n = X_sfe_n';
        [Y_sfe_n, sfe_output_ps] = mapminmax(Y_sfe_all', 0, 1);
        Y_sfe_n = Y_sfe_n';

        sfe_nets = cell(ENSEMBLE_N, 1);
        for e = 1:ENSEMBLE_N
            rng(3000 + e, 'twister');
            net = fitnet(hiddenLayerSize, trainFcn);
            net.divideFcn = 'dividetrain';
            net.trainParam.epochs = 2500;
            net.trainParam.goal = 1e-6;
            net.trainParam.min_grad = 1e-7;
            net.trainParam.showWindow = showTrainWindow;
            net.performFcn = 'mse';
            sfe_nets{e} = train(net, X_sfe_n', Y_sfe_n');
        end

        [SFE_fit, SFE_fit_std] = predictGEnsemble(sfe_nets, X_sfe_n, sfe_output_ps);
        R2_sfe_full = calcR2(Y_sfe_all, SFE_fit);
        fprintf('SFE 辅助集成模型（全数据）R^2 = %.4f，仅用于候选分散选点\n', R2_sfe_full);
        fprintf('SFE 辅助模型平均分歧 std = %.4f mJ/m^2\n', mean(SFE_fit_std));
        hasSFE_aux_model = true;
    else
        fprintf('\nSFE 有效样本不足，跳过 SFE 辅助模型；TopK 将仅按 G 与成分分散自动选点。\n');
    end
end



%% ===================== 8. GA 寻优（BP 集成均值 + 不确定性 + 适用域距离） ======================
fprintf('\n========== 开始 GA 寻优 (Target: Max ensemble G with uncertainty/domain control) ==========\n');

pop = zeros(POP_SIZE, 4);
for i = 1:POP_SIZE
    pop(i,:) = generateValidComp(RANGE);
end

best_score_history = zeros(GENS, 1);
best_G_history     = zeros(GENS, 1);
best_std_history   = zeros(GENS, 1);
best_dist_history  = zeros(GENS, 1);

global_best_score = -inf;
global_best_G     = -inf;
global_best_std   = nan;
global_best_dist  = nan;
global_best_comp  = [];

% 记录所有候选，用于最后输出候选池
all_comp_records  = [];
all_G_records     = [];
all_Gstd_records  = [];
all_D_records     = [];
all_SFE_records   = [];
all_SFEstd_records= [];
all_score_records = [];
all_gen_records   = [];

for gen = 1:GENS
    fitness    = zeros(POP_SIZE, 1);
    g_pred_vec = nan(POP_SIZE, 1);
    g_std_vec  = nan(POP_SIZE, 1);
    d_vec      = nan(POP_SIZE, 1);
    % SFE 预测向量也必须预先初始化为列向量，否则 MATLAB 会默认生成行向量，
    % 后续按 valid_gen 拼接时容易出现 vertcat 维度不一致。
    sfe_pred_vec = nan(POP_SIZE, 1);
    sfe_std_vec  = nan(POP_SIZE, 1);

    for i = 1:POP_SIZE
        comp = pop(i, :);

        % 约束检查
        [isValid, ~] = checkConstraints(comp, params, RANGE);

        if isValid
            x_in = buildFeatures(comp, best_mode_name, params);
            x_norm_col = mapminmax('apply', x_in', final_input_ps);
            x_norm = x_norm_col';

            [g_mu, g_std] = predictGEnsemble(final_nets, x_norm, final_output_ps);

            % 适用域距离：候选点到训练样本最近距离（归一化特征空间）
            d_train = min(sqrt(sum((X_best_n - x_norm).^2, 2)));

            % SFE 辅助预测：仅用于最终 TopK 自动分散选点，不进入 GA 个体适应度
            sfe_mu = NaN;
            sfe_std = NaN;
            if hasSFE_aux_model
                x_sfe_norm_col = mapminmax('apply', x_in', sfe_input_ps);
                x_sfe_norm = x_sfe_norm_col';
                [sfe_mu, sfe_std] = predictGEnsemble(sfe_nets, x_sfe_norm, sfe_output_ps);
            end

            % 适用域软惩罚：只惩罚明显远离训练数据的候选点
            penalty_unc  = LAMBDA_UNC  * g_std;
            penalty_dist = LAMBDA_DIST * max(0, d_train - DIST_TH)^2;

            score = g_mu - penalty_unc - penalty_dist;

            if ~isfinite(score)
                score = -1e12;
            end

            fitness(i)    = score;
            g_pred_vec(i) = g_mu;
            g_std_vec(i)  = g_std;
            d_vec(i)      = d_train;
            sfe_pred_vec(i) = sfe_mu;
            sfe_std_vec(i)  = sfe_std;
        else
            fitness(i)    = -1e12;
            g_pred_vec(i) = NaN;
            g_std_vec(i)  = NaN;
            d_vec(i)      = NaN;
            sfe_pred_vec(i) = NaN;
            sfe_std_vec(i)  = NaN;
        end
    end

    % 记录本代有效候选
    valid_gen = isfinite(g_pred_vec) & isfinite(fitness) & (fitness > -1e11);
    idx_gen = find(valid_gen);
    if ~isempty(idx_gen)
        all_comp_records   = [all_comp_records;   pop(idx_gen,:)];
        all_G_records      = [all_G_records;      g_pred_vec(idx_gen)];
        all_Gstd_records   = [all_Gstd_records;   g_std_vec(idx_gen)];
        all_D_records      = [all_D_records;      d_vec(idx_gen)];
        all_SFE_records    = [all_SFE_records;    sfe_pred_vec(idx_gen)];
        all_SFEstd_records = [all_SFEstd_records; sfe_std_vec(idx_gen)];
        all_score_records  = [all_score_records;  fitness(idx_gen)];
        all_gen_records    = [all_gen_records;    gen * ones(numel(idx_gen),1)];
    end

    % 更新全局最优
    [max_fit, max_idx] = max(fitness);
    if max_fit > global_best_score
        global_best_score = max_fit;
        global_best_comp  = pop(max_idx, :);
        global_best_G     = g_pred_vec(max_idx);
        global_best_std   = g_std_vec(max_idx);
        global_best_dist  = d_vec(max_idx);
    end

    best_score_history(gen) = global_best_score;
    best_G_history(gen)     = global_best_G;
    best_std_history(gen)   = global_best_std;
    best_dist_history(gen)  = global_best_dist;

    % ---------------- 轮盘赌选择（稳健版） ----------------
    bad_fit = ~isfinite(fitness);
    if any(bad_fit)
        finite_fit = fitness(isfinite(fitness));
        if isempty(finite_fit)
            fitness(:) = 1;
        else
            fitness(bad_fit) = min(finite_fit) - 1;
        end
    end

    fmin = min(fitness);
    fit_shift = fitness - fmin;
    fit_shift(~isfinite(fit_shift)) = 0;
    fit_shift(fit_shift < 0) = 0;

    if sum(fit_shift) <= 0
        probs = ones(POP_SIZE,1) / POP_SIZE;
    else
        probs = fit_shift / sum(fit_shift);
    end

    if any(~isfinite(probs)) || any(probs < 0) || sum(probs) <= 0
        probs = ones(POP_SIZE,1) / POP_SIZE;
    else
        probs = probs / sum(probs);
    end

    idx_sel = randsample(1:POP_SIZE, POP_SIZE, true, probs);
    idx_sel = idx_sel(:);

    pop = pop(idx_sel, :);
    new_pop = pop;

    % 精英保留：把当前全局最优直接保留到下一代第一位
    if ~isempty(global_best_comp)
        new_pop(1,:) = global_best_comp;
    end

    % ---------------- 交叉 ----------------
    for i = 2:2:POP_SIZE-1
        if rand < CROSS_RATE
            p1 = pop(i,:);
            p2 = pop(i+1,:);

            alpha = rand;
            c1 = alpha * p1 + (1 - alpha) * p2;
            c2 = (1 - alpha) * p1 + alpha * p2;

            new_pop(i,:)   = repairComp(c1, RANGE);
            new_pop(i+1,:) = repairComp(c2, RANGE);
        end
    end

    % ---------------- 变异 ----------------
    for i = 2:POP_SIZE   % 第1个位置保留精英
        if rand < MUT_RATE
            child = new_pop(i,:);
            mut_dim = randi(4);
            child(mut_dim) = child(mut_dim) + 0.08 * randn;
            child = repairComp(child, RANGE);

            [ok, ~] = checkConstraints(child, params, RANGE);
            if ok
                new_pop(i,:) = child;
            else
                new_pop(i,:) = generateValidComp(RANGE);
            end
        end
    end

    pop = new_pop;

    if mod(gen, 20) == 0 || gen == 1
        fprintf('Generation %d / %d, Best Score = %.4f | G_mu = %.4f GPa | std = %.4f | D = %.4f\n', ...
            gen, GENS, global_best_score, global_best_G, global_best_std, global_best_dist);
    end
end

%% ===================== 9. 输出最优结果 + Top 候选池 ======================
best_prop = calcPhysicalFeatures(global_best_comp, params);

fprintf('\n================= 单点最优结果（按 Score，仅作参考） =================\n');
fprintf('最佳特征组: %s\n', best_mode_name);
fprintf('最优组分 (at%%):\n');
fprintf('Fe: %.2f%%\n', global_best_comp(1) * 100);
fprintf('Cr: %.2f%%\n', global_best_comp(2) * 100);
fprintf('Ni: %.2f%%\n', global_best_comp(3) * 100);
fprintf('Mn: %.2f%%\n', global_best_comp(4) * 100);
fprintf('预测剪切模量 G 均值: %.4f GPa\n', global_best_G);
fprintf('集成模型预测 std: %.4f GPa\n', global_best_std);
fprintf('适用域距离 D: %.4f\n', global_best_dist);
fprintf('综合评分 Score: %.4f\n', global_best_score);

fprintf('\n对应物理参数:\n');
fprintf('VEC   = %.4f\n', best_prop.VEC);
fprintf('delta = %.4f\n', best_prop.delta);
fprintf('Hmix  = %.4f kJ/mol\n', best_prop.Hmix);
fprintf('Smix  = %.4f J/(mol·K)\n', best_prop.Smix);
fprintf('Omega = %.4f\n', best_prop.Omega);
fprintf('Tm_bar = %.4f K\n', best_prop.Tm_bar);

% -------- 构建候选池表 --------
candidate_table = table( ...
    all_comp_records(:,1)*100, ...
    all_comp_records(:,2)*100, ...
    all_comp_records(:,3)*100, ...
    all_comp_records(:,4)*100, ...
    all_G_records, ...
    all_Gstd_records, ...
    all_D_records, ...
    all_SFE_records, ...
    all_SFEstd_records, ...
    all_score_records, ...
    all_gen_records, ...
    'VariableNames', {'Fe_atpct','Cr_atpct','Ni_atpct','Mn_atpct', ...
                      'Predicted_G_mean_GPa','Predicted_G_std_GPa', ...
                      'AD_Distance','Predicted_SFE_mean_mJm2','Predicted_SFE_std_mJm2', ...
                      'Score','Generation'} );

% 按 Score 从大到小排序
candidate_table = sortrows(candidate_table, {'Score','Predicted_G_mean_GPa'}, {'descend','descend'});

% 从高 G 候选池中自动选取兼具成分/SFE分散性的 TopK
% 这一步是固定算法输出，不是人工挑选；最终 SFE 仍以 MD 复算为准。
top_table = selectDiverseTopCandidates(candidate_table, TOPK, POOLK, MIN_DIST, ...
    G_WINDOW, COMP_DIVERSITY_WEIGHT, SFE_DIVERSITY_WEIGHT);

% 给 top_table 加入物理特征列
delta_col = zeros(height(top_table),1);
Hmix_col  = zeros(height(top_table),1);
Smix_col  = zeros(height(top_table),1);
Omega_col = zeros(height(top_table),1);
VEC_col   = zeros(height(top_table),1);
Tm_col    = zeros(height(top_table),1);

for i = 1:height(top_table)
    c = [top_table.Fe_atpct(i), top_table.Cr_atpct(i), ...
         top_table.Ni_atpct(i), top_table.Mn_atpct(i)] / 100;
    p = calcPhysicalFeatures(c, params);
    delta_col(i) = p.delta;
    Hmix_col(i)  = p.Hmix;
    Smix_col(i)  = p.Smix;
    Omega_col(i) = p.Omega;
    VEC_col(i)   = p.VEC;
    Tm_col(i)    = p.Tm_bar;
end

top_table.delta  = delta_col;
top_table.Hmix   = Hmix_col;
top_table.Smix   = Smix_col;
top_table.Omega  = Omega_col;
top_table.VEC    = VEC_col;
top_table.Tm_bar = Tm_col;

disp(' ');
disp('================= Top 候选池（去重后） =================');
disp(top_table);

% 保存 CSV
writetable(top_table, 'Top_candidates_G_auto.csv');
fprintf('\nTop 候选池已保存为: Top_candidates_G_auto.csv\n');

%% ===================== 10. GA 收敛图 ======================
figure('Name','GA收敛历史','Color','w');
yyaxis left
plot(best_score_history, 'LineWidth', 2);
ylabel('Best Score');
yyaxis right
plot(best_G_history, '--', 'LineWidth', 2);
ylabel('Best Predicted G (GPa)');
xlabel('Generation');
title('GA Convergence History (ensemble G with uncertainty/domain control)');
grid on; box on;

%% ===================== 11. 最终 G 模型全数据拟合图 ======================
figure('Name','最终G模型全数据拟合','Color','w');
scatter(Y_best_all, Y_fit, 55, 'filled');
hold on;
xmin = min([Y_best_all; Y_fit]);
xmax = max([Y_best_all; Y_fit]);
plot([xmin xmax], [xmin xmax], 'k--', 'LineWidth', 1.5);
xlabel('Simulated G (GPa)');
ylabel('Predicted G (GPa)');
title(sprintf('Final G Model | Feature Set = %s | R^2 = %.4f', ...
    best_mode_name, R2_full));
grid on; box on;

%% ===================== 局部函数 ======================

function X = buildFeatures(comps, modeName, params)
    % 支持单个样本行向量 or 多个样本矩阵
    if isvector(comps)
        comps = comps(:)';
    end

    n = size(comps,1);
    phys = zeros(n, 6); % [delta, Hmix, Smix, Omega, VEC, Tm_bar]

    for i = 1:n
        p = calcPhysicalFeatures(comps(i,:), params);
        phys(i,:) = [p.delta, p.Hmix, p.Smix, p.Omega, p.VEC, p.Tm_bar];
    end

    switch lower(modeName)
        case 'comp3'
            X = comps(:,1:3);   % Fe Cr Ni
        case 'comp4'
            X = comps;          % Fe Cr Ni Mn
        case 'phys_only'
            X = phys;
        case 'comp3_phys'
            X = [comps(:,1:3), phys];
        case 'comp4_phys'
            X = [comps, phys];
        otherwise
            error('未知特征模式: %s', modeName);
    end
end

function prop = calcPhysicalFeatures(comp, params)
    comp = comp(:)';
    comp = comp / sum(comp);

    % -------- 1) 平均原子半径 --------
    r_bar = sum(comp .* params.r);

    % -------- 2) 原子尺寸差异 delta --------
    delta = sqrt(sum(comp .* (1 - params.r ./ r_bar).^2)) * 100;

    % -------- 3) 混合焓 Hmix --------
    Hmix = 0;
    for i = 1:4
        for j = i+1:4
            Hmix = Hmix + 4 * params.HmixPair(i,j) * comp(i) * comp(j);
        end
    end

    % -------- 4) 混合熵 Smix --------
    c_safe = comp;
    c_safe(c_safe < 1e-12) = 1e-12;
    Smix = -params.R * sum(c_safe .* log(c_safe));   % J/(mol*K)

    % -------- 5) 平均熔点 --------
    Tm_bar = sum(comp .* params.Tm);

    % -------- 6) 热力学参数 Omega --------
    if abs(Hmix) < 1e-8
        Omega = 1e6;
    else
        Omega = Tm_bar * Smix / abs(Hmix * 1000);
    end

    % -------- 7) VEC --------
    VEC = sum(comp .* params.VEC);

    prop.delta  = delta;
    prop.Hmix   = Hmix;
    prop.Smix   = Smix;
    prop.Omega  = Omega;
    prop.VEC    = VEC;
    prop.Tm_bar = Tm_bar;
end

function [isValid, prop] = checkConstraints(comp, params, range_lim)
    comp = comp(:)';
    comp = comp / sum(comp);

    prop = calcPhysicalFeatures(comp, params);

    range_ok = all(comp >= range_lim(1) - 1e-6) && all(comp <= range_lim(2) + 1e-6);
    sum_ok   = abs(sum(comp) - 1) < 1e-5;
    vec_ok   = prop.VEC >= 7.8;

    isValid = range_ok && sum_ok && vec_ok;
end

function comp = generateValidComp(range_lim)
    maxTry = 10000;
    for k = 1:maxTry
        r = rand(1,4);
        r = r / sum(r);

        if all(r >= range_lim(1)) && all(r <= range_lim(2))
            comp = r;
            return;
        end
    end
    error('generateValidComp: 在给定范围内长时间未生成有效组分。');
end

function comp_fixed = repairComp(comp, range_lim)
    comp = comp(:)';
    comp(comp < range_lim(1)) = range_lim(1);
    comp(comp > range_lim(2)) = range_lim(2);
    comp_fixed = comp / sum(comp);

    iter = 0;
    while (any(comp_fixed < range_lim(1)) || any(comp_fixed > range_lim(2))) && iter < 30
        comp_fixed(comp_fixed < range_lim(1)) = range_lim(1);
        comp_fixed(comp_fixed > range_lim(2)) = range_lim(2);
        comp_fixed = comp_fixed / sum(comp_fixed);
        iter = iter + 1;
    end

    if any(comp_fixed < range_lim(1)) || any(comp_fixed > range_lim(2))
        comp_fixed = generateValidComp(range_lim);
    end
end


function [y_mean, y_std] = predictGEnsemble(nets, X_norm, output_ps)
    % X_norm: nSample × nFeature, 已经按 final_input_ps 归一化
    nNet = numel(nets);
    n = size(X_norm, 1);
    preds = zeros(n, nNet);

    for e = 1:nNet
        y_n = nets{e}(X_norm');
        y   = mapminmax('reverse', y_n, output_ps)';
        preds(:, e) = y(:);
    end

    y_mean = mean(preds, 2);
    y_std  = std(preds, 0, 2);
end

function R2 = calcR2(y_true, y_pred)
    y_true = y_true(:);
    y_pred = y_pred(:);

    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);

    if ss_tot < 1e-12
        R2 = 0;
    else
        R2 = 1 - ss_res / ss_tot;
    end
end



function top_table = selectDiverseTopCandidates(candidate_table, TOPK, POOLK, min_dist, G_WINDOW, comp_w, sfe_w)
    % 自动 TopK 选择：高 G 候选池 + 成分空间分散
    % 这一步是固定规则，不是人工挑选。
    % 注意：最终 G/SFE 仍必须以 MD 复算为准。

    if isempty(candidate_table)
        top_table = candidate_table;
        return;
    end

    % 1) 基础清洗：去掉非法值
    need_cols = {'Fe_atpct','Cr_atpct','Ni_atpct','Mn_atpct','Score','Predicted_G_mean_GPa'};
    for k = 1:numel(need_cols)
        if ~ismember(need_cols{k}, candidate_table.Properties.VariableNames)
            error('candidate_table 缺少必要列: %s', need_cols{k});
        end
    end

    ok = isfinite(candidate_table.Score) & isfinite(candidate_table.Predicted_G_mean_GPa) & ...
         isfinite(candidate_table.Fe_atpct) & isfinite(candidate_table.Cr_atpct) & ...
         isfinite(candidate_table.Ni_atpct) & isfinite(candidate_table.Mn_atpct);
    candidate_table = candidate_table(ok,:);

    if isempty(candidate_table)
        top_table = candidate_table;
        return;
    end

    % 2) 按 Score/G 排序
    candidate_table = sortrows(candidate_table, {'Score','Predicted_G_mean_GPa'}, {'descend','descend'});

    % 3) 先做粗去重：把几乎相同的组分合并，避免重复行占据 TopK
    %    这里以 0.2 at.% 为网格进行唯一化，保留每个网格内排序最高者。
    comps_all = [candidate_table.Fe_atpct, candidate_table.Cr_atpct, ...
                 candidate_table.Ni_atpct, candidate_table.Mn_atpct];
    comp_key = round(comps_all / 0.2) * 0.2;
    [~, ia] = unique(comp_key, 'rows', 'stable');
    candidate_table = candidate_table(ia,:);

    % 4) 形成高 G 候选池：距最高预测 G 不超过 G_WINDOW 的所有点
    G_all = candidate_table.Predicted_G_mean_GPa;
    Gmax = max(G_all);
    high_idx = G_all >= (Gmax - G_WINDOW);
    pool = candidate_table(high_idx,:);

    % 若池太小，则退回排序前 POOLK 个
    if height(pool) < TOPK
        pool_n0 = min(POOLK, height(candidate_table));
        pool = candidate_table(1:pool_n0,:);
    else
        % 若池很大，只保留前 POOLK，保证速度
        pool_n0 = min(POOLK, height(pool));
        pool = pool(1:pool_n0,:);
    end

    pool_n = height(pool);
    comps = [pool.Fe_atpct, pool.Cr_atpct, pool.Ni_atpct, pool.Mn_atpct] / 100;
    Gval = pool.Predicted_G_mean_GPa;

    % 5) 归一化 G 和 Score，保证仍然优先高 G
    Gnorm = (Gval - min(Gval)) / (max(Gval) - min(Gval) + eps);
    ScoreVal = pool.Score;
    Snorm = (ScoreVal - min(ScoreVal)) / (max(ScoreVal) - min(ScoreVal) + eps);
    quality = 0.70 * Gnorm + 0.30 * Snorm;

    % 6) 若启用 SFE 辅助列，可作为可选分散项；默认权重可设为 0
    use_sfe = (sfe_w > 0) && ismember('Predicted_SFE_mean_mJm2', pool.Properties.VariableNames) && ...
              any(isfinite(pool.Predicted_SFE_mean_mJm2));
    if use_sfe
        SFE = pool.Predicted_SFE_mean_mJm2;
        finite_sfe = isfinite(SFE);
        if sum(finite_sfe) >= TOPK
            SFE(~finite_sfe) = median(SFE(finite_sfe));
            SFEnorm = (SFE - min(SFE)) / (max(SFE) - min(SFE) + eps);
        else
            use_sfe = false;
            SFEnorm = zeros(pool_n,1);
            sfe_w = 0;
        end
    else
        SFEnorm = zeros(pool_n,1);
        sfe_w = 0;
    end

    selected_idx = [];

    % 7) 第一个点选质量最高者
    [~, first_idx] = max(quality);
    selected_idx(1) = first_idx;

    % 8) 后续点采用固定的质量-多样性准则自动选择
    %    如果 min_dist 太严格，逐步放宽，但不会退化成重复点。
    current_min_dist = min_dist;

    while numel(selected_idx) < min(TOPK, pool_n)
        best_score = -inf;
        best_i = -1;

        for i = 1:pool_n
            if any(selected_idx == i)
                continue;
            end

            c = comps(i,:);
            selected_comps = comps(selected_idx,:);
            d = sqrt(sum((selected_comps - c).^2, 2));
            min_comp_d = min(d);

            if min_comp_d < current_min_dist
                continue;
            end

            if use_sfe
                sfe_d = abs(SFEnorm(selected_idx) - SFEnorm(i));
                min_sfe_d = min(sfe_d);
            else
                min_sfe_d = 0;
            end

            % 归一化距离，避免距离量纲太小导致不起作用
            div_score = min(min_comp_d / max(min_dist, eps), 2.0) / 2.0;
            final_score = quality(i) + comp_w * div_score + sfe_w * min_sfe_d;

            if final_score > best_score
                best_score = final_score;
                best_i = i;
            end
        end

        % 如果找不到满足当前距离的候选，逐步放宽距离阈值
        if best_i < 0
            current_min_dist = current_min_dist * 0.85;
            if current_min_dist < 0.03
                % 最后兜底：直接选与已选集合距离最大的高质量点
                best_score = -inf;
                for i = 1:pool_n
                    if any(selected_idx == i)
                        continue;
                    end
                    c = comps(i,:);
                    selected_comps = comps(selected_idx,:);
                    d = sqrt(sum((selected_comps - c).^2, 2));
                    min_comp_d = min(d);
                    div_score = min_comp_d;
                    final_score = quality(i) + comp_w * div_score;
                    if final_score > best_score
                        best_score = final_score;
                        best_i = i;
                    end
                end
                selected_idx(end+1) = best_i;
            end
        else
            selected_idx(end+1) = best_i;
        end
    end

    top_table = pool(selected_idx,:);

    % 9) 附加输出：与已选候选的最近成分距离，便于检查是否仍过于相似
    top_comps = [top_table.Fe_atpct, top_table.Cr_atpct, top_table.Ni_atpct, top_table.Mn_atpct] / 100;
    nearestD = nan(height(top_table),1);
    for i = 1:height(top_table)
        if height(top_table) == 1
            nearestD(i) = nan;
        else
            others = top_comps;
            others(i,:) = [];
            nearestD(i) = min(sqrt(sum((others - top_comps(i,:)).^2,2)));
        end
    end
    top_table.NearestCompDist = nearestD;
end
