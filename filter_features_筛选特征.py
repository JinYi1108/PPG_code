import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from eli5.sklearn import PermutationImportance
import json

warnings.filterwarnings("ignore")
import shap
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, SGDClassifier
import sys

from boruta import BorutaPy


TRAIN_FILE_PATH = '1111.xlsx'
TEST_A_PATH = 'a.xlsx'
TEST_B_PATH = 'b.xlsx'
TEST_C_PATH = 'c.xlsx'
NOMOGRAM_ROC_DATA_PATH = 'nomogram_roc_curve_data.csv'

TARGET_VARIABLE = 'PPG'
RANDOM_STATE = 42
N_SPLITS_CV = 5 


print("--- 步骤1: 加载数据 ---")
train_df = pd.read_excel(TRAIN_FILE_PATH)
combined_test_df = pd.concat([pd.read_excel(p) for p in [TEST_A_PATH, TEST_B_PATH, TEST_C_PATH]], ignore_index=True)
X_train_full = train_df.drop(columns=[TARGET_VARIABLE])
y_train_full = train_df[TARGET_VARIABLE]
X_test_combined = combined_test_df.drop(columns=[TARGET_VARIABLE])
y_test_combined = combined_test_df[TARGET_VARIABLE]

scaler = StandardScaler()
X_train_scaled_df = pd.DataFrame(scaler.fit_transform(X_train_full), columns=X_train_full.columns)
X_test_scaled_df = pd.DataFrame(scaler.transform(X_test_combined), columns=X_test_combined.columns)
print("数据加载和准备完成。")


print("\n--- 步骤2: 定义模型库和超参数网格 ---")
counts = y_train_full.value_counts()
scale_pos_weight_value = counts[0] / counts[1]


base_models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=2000),
    'Elastic Net': SGDClassifier(loss='log_loss', penalty='elasticnet', random_state=RANDOM_STATE, class_weight='balanced', max_iter=2000),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
    'SVM': SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
    'LightGBM': lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, scale_pos_weight=scale_pos_weight_value),
    'XGBoost': xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss', scale_pos_weight=scale_pos_weight_value)
}


param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Elastic Net': {'alpha': [0.01, 0.1], 'l1_ratio': [0.15, 0.85]},
    'Decision Tree': { 
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],
    'criterion': ['gini', 'entropy']},

    'Random Forest': { 
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [10, 40],
    'min_samples_leaf': [5, 10],
    'ccp_alpha': [0.0, 0.01]},

    'SVM': {  
    'C': [0.1, 0.15, 1],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']},

    'LightGBM': { 
    'n_estimators': [100,200],
    'learning_rate': [0.01,0.05],
    'num_leaves': [15,30],
    'max_depth': [-1],
    'min_child_samples': [30,50],
    'subsample': [0.7,1.0],
    'colsample_bytree': [0.7,1.0],
    "reg_alpha":[0.1,0.3]},

    'XGBoost': {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 150],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    "gamma": [0, 0.1],
    "reg_alpha": [0, 0.1]
    }
}
print("模型库和超参数网格定义完成。")

print("\n--- 步骤3: 运行 Boruta 算法进行特征选择 ---")
rf_boruta = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta_selector = BorutaPy(rf_boruta, n_estimators='auto', verbose=0, random_state=RANDOM_STATE)
boruta_selector.fit(X_train_full.values, y_train_full.values)
boruta_confirmed_features = X_train_full.columns[boruta_selector.support_].tolist()
print(f"Boruta 确认了 {len(boruta_confirmed_features)} 个重要特征: \n{boruta_confirmed_features}")


print("\n--- 步骤4: 为每个模型独立寻找最佳特征集与超参数 ---")
ultimate_results_list = []
all_rfe_curves_data = []
for name, model in base_models.items():
    print(f"\n========== 正在处理模型: {name} ==========")
    
    is_tree_model = name in ['Decision Tree', 'Random Forest', 'LightGBM', 'XGBoost']
    

    print(f"  a) 获取 {name} 的专属特征排名")
    X_train_temp = X_train_full if is_tree_model else X_train_scaled_df


    print(f"     训练 {name} 以获取重要性")
    model.fit(X_train_temp, y_train_full)


    if name in ['LightGBM', 'XGBoost', 'Random Forest', 'Decision Tree']:
        print("     使用 SHAP TreeExplainer 获取特征重要性")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_temp)
        if isinstance(shap_values, list):

            shap_values_for_class_1 = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:

            shap_values_for_class_1 = shap_values[:, :, 1]
        else:

            shap_values_for_class_1 = shap_values
    

        importances = np.mean(np.abs(shap_values_for_class_1), axis=0)

    elif hasattr(model, 'coef_'):
        print("     使用模型系数 (coef_) 获取特征重要性")
        importances = np.abs(model.coef_[0])

    else: 
        print("     (模型无内置重要性，改用置换重要性)")
        perm = PermutationImportance(model, scoring='roc_auc', n_iter=5, random_state=RANDOM_STATE).fit(X_train_temp, y_train_full)
        importances = perm.feature_importances_


    print(f"     - 原始 importances 类型: {type(importances)}, 形状: {getattr(importances, 'shape', 'N/A')}")


    importances_flat = np.array(importances).flatten()
    if len(importances_flat) != X_train_full.shape[1]:
        raise ValueError(f"模型 '{name}' 的重要性计算结果长度 ({len(importances_flat)}) "
                     f"与特征数量 ({X_train_full.shape[1]}) 不匹配。")


    ranked_features = X_train_full.columns.to_numpy()[np.argsort(importances_flat)[::-1]].tolist()

    # print(f"  a) 获取 {name} 的专属特征排名")
    # X_train_temp = X_train_full if is_tree_model else X_train_scaled_df
    # model.fit(X_train_temp, y_train_full)
    # if name in ['LightGBM', 'XGBoost', 'Random Forest', 'Decision Tree']:


    #     print("     使用 SHAP TreeExplainer 获取特征重要性")
    #     explainer = shap.TreeExplainer(model)
    #     shap_values = explainer.shap_values(X_train_temp)

    #     if isinstance(shap_values, list):
    #         shap_values_for_class_1 = shap_values[1]
    #     else: 
    #         shap_values_for_class_1 = shap_values
    

    #     importances = np.mean(np.abs(shap_values_for_class_1), axis=0)

    # elif hasattr(model, 'coef_'):


    #     print("     使用模型系数 (coef_) 获取特征重要性")
    #     importances = np.abs(model.coef_[0])

    # else: 


    #     print("     (模型无内置重要性，改用置换重要性)")
    #     perm = PermutationImportance(model, scoring='roc_auc', n_iter=5, random_state=RANDOM_STATE).fit(X_train_temp, y_train_full)
    #     importances = perm.feature_importances_

    
    
    ranked_features = X_train_full.columns.to_numpy()[np.argsort(importances)[::-1]].tolist()


    print(f"  b) 为 {name} 进行递归特征削减与超参数搜索")
    rfe_grid_results = []
    for n in range(len(ranked_features), 0, -1):
        top_n_features = ranked_features[:n]
        X_train_subset = X_train_full[top_n_features]
        X_train_scaled_subset = X_train_scaled_df[top_n_features]
        
        X_tr_subset = X_train_subset if is_tree_model else X_train_scaled_subset
        

        grid_search = GridSearchCV(model, param_grids[name], cv=N_SPLITS_CV, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_tr_subset, y_train_full)
        
        rfe_grid_results.append({
            'num_features': n,
            'best_cv_auc': grid_search.best_score_,
            'best_params': grid_search.best_params_
        })
        print(f"    {n} 个特征下的最佳CV AUC: {grid_search.best_score_:.4f}")


    best_config_df = pd.DataFrame(rfe_grid_results)
    

    MIN_FEATURES_LIMIT = 5  
    TOLERANCE = 0.99      

    ABSOLUTE_AUC_THRESHOLD = 0.9

    max_auc = best_config_df['best_cv_auc'].max()

 
    tolerance_threshold = max_auc * TOLERANCE
    

    if max_auc >= ABSOLUTE_AUC_THRESHOLD:

    
        
        candidate_configs = best_config_df[
            (best_config_df['best_cv_auc'] >= ABSOLUTE_AUC_THRESHOLD) & 
            (best_config_df['num_features'] >= MIN_FEATURES_LIMIT)
        ]
        
    else:
    
    
        
        candidate_configs = best_config_df[
            (best_config_df['best_cv_auc'] >= tolerance_threshold) & 
            (best_config_df['num_features'] >= MIN_FEATURES_LIMIT)
        ]

    if not candidate_configs.empty:
        best_run = candidate_configs.loc[candidate_configs['num_features'].idxmin()]
        decision_method = "最优选择 (性能>99% & 特征>=5)"
    else:

        best_run = best_config_df.loc[best_config_df['best_cv_auc'].idxmax()]
        decision_method = "备选方案 (仅选择最高AUC)"


    print(f"    - 决策方法: {decision_method}")
    print(f"    - 最佳性能 (Max CV AUC): {max_auc:.4f}")
    print(f"    - 性能容忍阈值 (>= {TOLERANCE*100}%): {tolerance_threshold:.4f}")
    print(f"    - 最终选择: {int(best_run['num_features'])} 个特征时, CV AUC 达到 {best_run['best_cv_auc']:.4f}")


    ultimate_results_list.append({
        'Model': name,
        'Best_Num_Features': int(best_run['num_features']),
        'Best_CV_AUC': best_run['best_cv_auc'],
        'Best_Params': best_run['best_params'],
        'Best_Feature_Set': ranked_features[:int(best_run['num_features'])]
    })
    best_config_df['Model'] = name
    all_rfe_curves_data.append(best_config_df)

    


print("\n\n--- 步骤5: 汇总所有模型的最佳配置 ---")
ultimate_summary_df = pd.DataFrame(ultimate_results_list).sort_values('Best_CV_AUC', ascending=False)

pd.set_option('display.max_colwidth', None)
print(ultimate_summary_df[['Model', 'Best_Num_Features', 'Best_CV_AUC']])
print(ultimate_summary_df[['Model', 'Best_Num_Features', 'Best_Params']])
configs_to_save = ultimate_summary_df.set_index('Model')[['Best_Feature_Set', 'Best_Params']].rename(
    columns={'Best_Feature_Set': 'features', 'Best_Params': 'params'}
).to_dict(orient='index')


with open('best_model_configs_new.json', 'w', encoding='utf-8') as f:
    json.dump(configs_to_save, f, indent=4, ensure_ascii=False)

print("\n所有模型的最佳配置已保存到 'best_model_configs.json' 文件中")



print(json.dumps(configs_to_save.get('XGBoost'), indent=4))

print("\n\n--- 步骤6: 绘制综合性能曲线对比图 ---")

all_curves_df = pd.concat(all_rfe_curves_data, ignore_index=True)

plt.figure(figsize=(16, 9))
sns.lineplot(data=all_curves_df, x='num_features', y='best_cv_auc', hue='Model', marker='o', palette='tab10')
plt.title('Performance vs. Number of Features for Each Model', fontsize=16)
plt.xlabel('Number of Features Used (Model-Specific Ranking)', fontsize=12)
plt.ylabel(f'Best Mean {N_SPLITS_CV}-Fold CV AUC', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().invert_xaxis() 
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



