import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import joblib

warnings.filterwarnings("ignore")
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score
from dcurves import dca
from scipy.stats import chi2
from scipy.stats import chi2, norm

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, SGDClassifier





CONFIG_FILE_PATH = 'best_model_configs.json'
TRAIN_FILE_PATH = '1111.xlsx'
TEST_A_PATH = 'a.xlsx'
TEST_B_PATH = 'b.xlsx'
TEST_C_PATH = 'c.xlsx'
NOMOGRAM_DCA_CIC_DATA_PATH = 'nomogram_DCA_data.csv'
NOMOGRAM_ROC_DATA_PATH = 'nomogram_roc_curve_data.csv'
NOMOGRAM_METRICS_PATH = 'nomogram_performance_metrics_full.csv'

TARGET_VARIABLE = 'PPG'
RANDOM_STATE = 42



def hosmer_lemeshow_test(y_true, y_pred_proba, n_groups=10):
    
    y_true, y_pred_proba = np.array(y_true), np.array(y_pred_proba)
    data = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba}).sort_values('y_pred_proba')
    try:
        data['group'] = pd.qcut(data['y_pred_proba'], n_groups, duplicates='raise', labels=False)
    except ValueError:
        n_groups = max(2, int(len(data) / 10))
        if n_groups < 2: return np.nan, np.nan
        data['group'] = pd.qcut(data['y_pred_proba'].rank(method='first'), n_groups, duplicates='drop', labels=False)
    
    observed = data.groupby('group')['y_true'].sum()
    expected = data.groupby('group')['y_pred_proba'].sum()
    n_obs = data.groupby('group')['y_true'].count()
    
    observed_non_event = n_obs - observed
    expected_non_event = n_obs - expected
    
    chi_sq_stat = ((observed - expected)**2 / (expected + 1e-9)).sum() + \
                  ((observed_non_event - expected_non_event)**2 / (expected_non_event + 1e-9)).sum()
    p_value = 1 - chi2.cdf(chi_sq_stat, max(1, n_groups - 2))
    return chi_sq_stat, p_value


def compute_midrank(x):

    sorted_x = np.sort(x)
    unique_x, inverse, counts = np.unique(sorted_x, return_inverse=True, return_counts=True)
    midranks = np.cumsum(counts) - (counts - 1) / 2.0
    return midranks[inverse[np.argsort(np.argsort(x))]] 

def delong_roc_variance(ground_truth, predictions):

    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)
    
    
    positive_pred = predictions[ground_truth == 1]
    negative_pred = predictions[ground_truth == 0]
    
    n_pos = len(positive_pred)
    n_neg = len(negative_pred)
    
    if n_pos == 0 or n_neg == 0:
        return 0, 0 

    v10 = np.zeros(n_pos)
    for i in range(n_pos):
        v10[i] = (negative_pred < positive_pred[i]).mean() + (negative_pred == positive_pred[i]).mean() * 0.5

    v01 = np.zeros(n_neg)
    for i in range(n_neg):
        v01[i] = (positive_pred > negative_pred[i]).mean() + (positive_pred == negative_pred[i]).mean() * 0.5

    auc = np.mean(v10)
    var_auc = (np.var(v10, ddof=1) / n_pos) + (np.var(v01, ddof=1) / n_neg)
    
    return auc, var_auc

def delong_roc_test(y_true, y_pred1, y_pred2):

    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return np.nan 


    positive_pred1 = y_pred1[y_true == 1]
    negative_pred1 = y_pred1[y_true == 0]
    positive_pred2 = y_pred2[y_true == 1]
    negative_pred2 = y_pred2[y_true == 0]


    v10 = np.zeros(n_pos)
    v01 = np.zeros(n_neg)
    v20 = np.zeros(n_pos)
    v02 = np.zeros(n_neg)

    for i in range(n_pos):
        v10[i] = (negative_pred1 < positive_pred1[i]).mean() + 0.5 * (negative_pred1 == positive_pred1[i]).mean()
        v20[i] = (negative_pred2 < positive_pred2[i]).mean() + 0.5 * (negative_pred2 == positive_pred2[i]).mean()

    for i in range(n_neg):
        v01[i] = (positive_pred1 > negative_pred1[i]).mean() + 0.5 * (positive_pred1 == negative_pred1[i]).mean()
        v02[i] = (positive_pred2 > negative_pred2[i]).mean() + 0.5 * (positive_pred2 == negative_pred2[i]).mean()
        
    auc1 = np.mean(v10)
    auc2 = np.mean(v20)


    var1 = (np.var(v10, ddof=1) / n_pos) + (np.var(v01, ddof=1) / n_neg)
    var2 = (np.var(v20, ddof=1) / n_pos) + (np.var(v02, ddof=1) / n_neg)
    

    cov_v10_v20 = np.cov(v10, v20, ddof=1)[0, 1]
    cov_v01_v02 = np.cov(v01, v02, ddof=1)[0, 1]
    
    cov = (cov_v10_v20 / n_pos) + (cov_v01_v02 / n_neg)
    

    se = np.sqrt(var1 + var2 - 2 * cov)
    if se == 0: return 1.0
    
    z = (auc1 - auc2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    
    return auc1, auc2, p

def calculate_clinical_impact(data_with_preds, outcome_col, model_names, thresholds):

    results = []
    total_patients = len(data_with_preds)
    
    for threshold in thresholds:
        for model in model_names:

            high_risk_mask = data_with_preds[model] >= threshold
            n_high_risk = high_risk_mask.sum()
            

            high_risk_with_outcome_mask = (high_risk_mask) & (data_with_preds[outcome_col] == 1)
            n_high_risk_with_outcome = high_risk_with_outcome_mask.sum()
            
            results.append({
                "model": model,
                "threshold": threshold,
                "n_high_risk": n_high_risk,
                "n_high_risk_with_outcome": n_high_risk_with_outcome
            })
            
    return pd.DataFrame(results)



def calculate_metrics_with_bootstrap(y_true, y_pred_proba, threshold=0.5, n_bootstrap=1000, seed=42):


    boot_metrics = []
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred_label = (y_pred_proba >= threshold).astype(int)


    metrics_orig = {
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Accuracy': accuracy_score(y_true, y_pred_label),
        'Recall': recall_score(y_true, y_pred_label),
        'Precision': precision_score(y_true, y_pred_label),
        'F1': f1_score(y_true, y_pred_label)
    }


    rng = np.random.RandomState(seed)
    for i in range(n_bootstrap):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue 
            
        y_true_boot = y_true[indices]
        y_proba_boot = y_pred_proba[indices]
        y_label_boot = (y_proba_boot >= threshold).astype(int)

        boot_metrics.append([
            roc_auc_score(y_true_boot, y_proba_boot),
            accuracy_score(y_true_boot, y_label_boot),
            recall_score(y_true_boot, y_label_boot),
            precision_score(y_true_boot, y_label_boot),
            f1_score(y_true_boot, y_label_boot)
        ])

    boot_metrics = np.array(boot_metrics)
    

    lower_bounds = np.percentile(boot_metrics, 2.5, axis=0)
    upper_bounds = np.percentile(boot_metrics, 97.5, axis=0)


    results_formatted = {}
    metric_names = ['AUC', 'Accuracy', 'Recall', 'Precision', 'F1']
    for idx, name in enumerate(metric_names):
        results_formatted[name] = f"{metrics_orig[name]:.3f} ({lower_bounds[idx]:.3f}-{upper_bounds[idx]:.3f})"
    
    return results_formatted

print("--- 步骤1: 加载数据集 ---")

all_datasets = {
    "Train Set": pd.read_excel(TRAIN_FILE_PATH),
    "Test Set A": pd.read_excel(TEST_A_PATH),
    "Test Set B": pd.read_excel(TEST_B_PATH),
    "Test Set C": pd.read_excel(TEST_C_PATH)
}
all_datasets["Combined Test Set"] = pd.concat([all_datasets["Test Set A"], all_datasets["Test Set B"], all_datasets["Test Set C"]], ignore_index=True)
print("所有数据集加载成功。")


X_train_full = all_datasets["Train Set"].drop(columns=[TARGET_VARIABLE])
y_train_full = all_datasets["Train Set"][TARGET_VARIABLE]



print("\n--- 步骤2: 加载最佳模型配置 ---")
with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
    best_configs = json.load(f)


print("\n--- 步骤3: 在所有数据集上进行模型评估 ---")
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

all_results = []

all_roc_data = {}
all_dca_data = {}

for model_name, config in best_configs.items():
    print(f"\n========== 正在评估模型: {model_name} ==========")
    features, params = config['features'], config['params']
    X_train_subset = X_train_full[features]
    

    base_model = base_models[model_name].set_params(**params)
    

    is_tree_model = model_name in ['Decision Tree', 'Random Forest', 'LightGBM', 'XGBoost']
    if not is_tree_model:
        scaler = StandardScaler()
        X_train_subset_scaled = scaler.fit_transform(X_train_subset)
    else:
        X_train_subset_scaled = X_train_subset 
        

    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    

    if not is_tree_model:
        calibrated_model.fit(X_train_subset_scaled, y_train_full)
    else:
        calibrated_model.fit(X_train_subset, y_train_full)
    
    if model_name == 'XGBoost':
        print(f"--- 正在保存 {model_name} (校准后) 的模型 ---")
        joblib.dump(calibrated_model, 'xgb_calibrated_model.pkl')
        print("  -> 'xgb_calibrated_model.pkl' 已保存。")
        
    for cohort_name, cohort_df in all_datasets.items():
        X_eval = cohort_df[features]
        y_eval = cohort_df[TARGET_VARIABLE]
        
        if not is_tree_model:
            X_eval_scaled = scaler.transform(X_eval)
            y_pred_proba = calibrated_model.predict_proba(X_eval_scaled)[:, 1]
        else:
            y_pred_proba = calibrated_model.predict_proba(X_eval)[:, 1]
        print(cohort_name,cohort_df,y_pred_proba)


        print(f"  计算 {cohort_name} 的 Bootstrap 置信区间...")
        metrics_ci = calculate_metrics_with_bootstrap(y_eval, y_pred_proba, n_bootstrap=1000)
        
        _, hl_p_value = hosmer_lemeshow_test(y_eval, y_pred_proba)
        
        all_results.append({
            'Model': model_name, 
            'Cohort': cohort_name, 
            'AUC_95CI': metrics_ci['AUC'],
            'HL_p_value': hl_p_value, 
            'Accuracy_95CI': metrics_ci['Accuracy'],
            'Recall_95CI': metrics_ci['Recall'], 
            'Precision_95CI': metrics_ci['Precision'],
            'F1-Score_95CI': metrics_ci['F1']
        })

        _, hl_p_value = hosmer_lemeshow_test(y_eval, y_pred_proba)
        all_results.append({
            'Model': model_name, 'Cohort': cohort_name, 'AUC': roc_auc_score(y_eval, y_pred_proba),
            'HL_p_value': hl_p_value, 'Accuracy': accuracy_score(y_eval, (y_pred_proba >= 0.5)),
            'Recall': recall_score(y_eval, (y_pred_proba >= 0.5)), 'Precision': precision_score(y_eval, (y_pred_proba >= 0.5)),
            'F1-Score': f1_score(y_eval, (y_pred_proba >= 0.5))
        })
        

        fpr, tpr, _ = roc_curve(y_eval, y_pred_proba)
        if cohort_name not in all_roc_data: all_roc_data[cohort_name] = {}
        all_roc_data[cohort_name][model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc_score(y_eval, y_pred_proba)}
        
        if cohort_name not in all_dca_data: all_dca_data[cohort_name] = pd.DataFrame({'PPG': y_eval})
        all_dca_data[cohort_name][model_name] = y_pred_proba


print("\n\n--- 步骤4: 汇总与展示最终性能 ---")
results_df = pd.DataFrame(all_results)
results_df.to_excel('各模型各集合上性能.xlsx', index=False)
summary_table = results_df.pivot_table(
    index='Model', 
    columns='Cohort', 
    values=['AUC', 'HL_p_value']
).sort_values(by=('AUC', 'Combined Test Set'), ascending=False)

print("\n模型性能综合对比表 (AUC & H-L p-value):")
print(summary_table.round(3)) 




print("\n--- 步骤5: 在合并测试集上进行可视化对比 ---")
cohort_to_plot =  "Test Set A"

#  "Train Set": pd.read_excel(TRAIN_FILE_PATH),
#     "Test Set A": pd.read_excel(TEST_A_PATH),
#     "Test Set B": pd.read_excel(TEST_B_PATH),
#     "Test Set C": pd.read_excel(TEST_C_PATH)
# }
# all_datasets["Combined Test Set"]


plt.figure(figsize=(10, 8))
for model_name, data in all_roc_data[cohort_to_plot].items():
    plt.plot(data['fpr'], data['tpr'], label=f"{model_name} (AUC = {data['auc']:.3f})")
try:
    nomo_roc_df = pd.read_csv(NOMOGRAM_ROC_DATA_PATH)
    nomo_combined = nomo_roc_df[nomo_roc_df['Cohort'] == 'Test_Combined']
    nomo_metrics_df = pd.read_csv(NOMOGRAM_METRICS_PATH)
    nomo_auc = nomo_metrics_df[nomo_metrics_df['Cohort'] == 'Test Set (Combined)']['AUC'].iloc[0]
    plt.plot(nomo_combined['FPR'], nomo_combined['TPR'], label=f"Nomogram (AUC = {nomo_auc:.3f})", linestyle='--', c='black', lw=2.5)
except Exception: pass
plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC Curve Comparison on {cohort_to_plot}'); plt.legend(); plt.grid(); plt.show()

print(f"  为 {cohort_to_plot} 绘制DCA曲线...")
dca_df_to_plot = all_dca_data[cohort_to_plot]
model_names_for_dca = list(best_configs.keys())



dca_results_df = dca(
    data=dca_df_to_plot,
    outcome='PPG',
    modelnames=model_names_for_dca
)

try:
    nomo_dca_full_df = pd.read_csv(NOMOGRAM_DCA_CIC_DATA_PATH)
    if 'model' not in nomo_dca_full_df.columns and 'variable' in nomo_dca_full_df.columns:
        nomo_dca_full_df = nomo_dca_full_df.rename(columns={'variable': 'model'})
    nomo_dca_combined = nomo_dca_full_df[
        (nomo_dca_full_df['Cohort'] == "Combined Test Set") & 
        (nomo_dca_full_df['model'] == 'Nomogram') 
    ]
    
    dca_combined_plot_df = pd.concat([dca_results_df, nomo_dca_combined])
    print("  加载并合并Nomogram的DCA数据。")
except Exception as e:
    print(f"加载Nomogram DCA数据失败: {e}。DCA图将只显示ML模型。")
    dca_combined_plot_df = dca_results_df

plt.figure(figsize=(10, 8))

sns.lineplot(data=dca_combined_plot_df, x='threshold', y='net_benefit', hue='model', linewidth=2)
plt.ylim(-0.1, 0.6)
plt.title(f'Decision Curve Analysis on {cohort_to_plot}')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Models')
plt.show()






print(f"\n--- 步骤6: DeLong's Test (在 {cohort_to_plot} 上) ---")
try:
    best_ml_model_name = summary_table[('AUC', 'Combined Test Set')].idxmax()
    print(f"  最佳ML模型 ({best_ml_model_name}) vs Nomogram (Logistic Regression)")
    
    y_true_combined = all_datasets[cohort_to_plot][TARGET_VARIABLE]
    y_pred_best_ml = all_dca_data[cohort_to_plot][best_ml_model_name]
    y_pred_nomogram = all_dca_data[cohort_to_plot]['Logistic Regression']
    

    auc1, auc2, p_value = delong_roc_test(y_true_combined, y_pred_best_ml, y_pred_nomogram)
    
    print(f"\n  DeLong's Test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  {best_ml_model_name} (AUC={auc1:.3f}) 与 Nomogram (AUC={auc2:.3f}) 之间存在统计学上的显著差异。")
    else:
        print(f"  {best_ml_model_name} (AUC={auc1:.3f}) 与 Nomogram (AUC={auc2:.3f}) 之间没有统计学上的显著差异。")
        
except Exception as e:
    print(f"  DeLong's Test 执行失败: {e}")








print(f"\n--- 步骤7: 绘制临床影响曲线 (CIC) (在 {cohort_to_plot} 上) ---")
try:

    cic_df = calculate_clinical_impact(
        data_with_preds=all_dca_data[cohort_to_plot],
        outcome_col='PPG',
        model_names=list(best_configs.keys()),
        thresholds=np.linspace(0.01, 0.99, 100)
    )
    

    cic_df_long = cic_df.melt(
        id_vars=['model', 'threshold'], 
        value_vars=['n_high_risk', 'n_high_risk_with_outcome'],
        var_name='Curve_Type', 
        value_name='Number_of_Patients'
    )

    NOMOGRAM_CIC_DATA_PATH = "nomogram_CIC_data.csv"
    nomo_cic_df = pd.read_csv(NOMOGRAM_CIC_DATA_PATH)

    if 'model' not in nomo_cic_df.columns and 'variable' in nomo_cic_df.columns:
        nomo_cic_df = nomo_cic_df.rename(columns={'variable': 'model'})

    nomo_cic_filtered = nomo_cic_df[
        (nomo_cic_df['Cohort'] == "Combined Test Set") & 
        (nomo_cic_df['model'] == 'Nomogram')
    ].copy() 
    
    nomo_cic_filtered = nomo_cic_filtered.rename(columns={
        'n_risk': 'n_high_risk',
        'n_risk_pos': 'n_high_risk_with_outcome'
    })
    nomo_cic_long_df = nomo_cic_filtered.melt(
        id_vars=['model', 'threshold'], 
        value_vars=['n_high_risk', 'n_high_risk_with_outcome'], 
        var_name='Curve_Type', 
        value_name='Number_of_Patients'
    )

    cic_combined_plot_df = pd.concat([cic_df_long, nomo_cic_long_df])
    # nomo_dca_full_df = pd.read_csv(NOMOGRAM_DCA_CIC_DATA_PATH)
    # if 'model' not in nomo_dca_full_df.columns and 'variable' in nomo_dca_full_df.columns:
    #     nomo_dca_full_df = nomo_dca_full_df.rename(columns={'variable': 'model'})
    
    # nomo_cic_combined = nomo_dca_full_df[(nomo_dca_full_df['Cohort'] == cohort_to_plot) & (nomo_dca_full_df['model'] == 'Nomogram')]
    # nomo_cic_long_df = nomo_cic_combined.melt(id_vars=['model', 'threshold'], value_vars=['n_high_risk', 'n_high_risk_with_outcome'], var_name='Curve_Type', value_name='Number_of_Patients')
    # cic_combined_plot_df = pd.concat([cic_df_long, nomo_cic_long_df])
    # cic_combined_plot_df['Curve_Type'] = cic_combined_plot_df['Curve_Type'].map({'n_high_risk': 'Number High Risk', 'n_high_risk_with_outcome': 'Number High Risk with Event'})

    cic_combined_plot_df['Curve_Type'] = cic_combined_plot_df['Curve_Type'].map({
        'n_high_risk': 'Number High Risk', 
        'n_high_risk_with_outcome': 'Number High Risk with Event'
    })


    plt.figure(figsize=(12, 8))
    sns.lineplot(data=cic_combined_plot_df, 
                 x='threshold', 
                 y='Number_of_Patients', 
                 hue='model',        
                 style='Curve_Type', 
                 linewidth=2)
    
    plt.title(f'Clinical Impact Curve on {cohort_to_plot}')
    plt.xlabel('High Risk Threshold')
    plt.ylabel('Number of Patients')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

except Exception as e:
    print(f"  CIC 绘图失败: {e}")

print("\n--- 评估流程结束 ---")



















print(f"\n--- 步骤8: 绘制 ROC 曲线 (XGBoost vs. Nomogram) ---")
plt.figure(figsize=(10, 8))
try:
    xgb_roc_data = all_roc_data[cohort_to_plot]['XGBoost']
    auc_xgb = xgb_roc_data['auc']
    plt.plot(xgb_roc_data['fpr'], xgb_roc_data['tpr'], label=f"XGBoost (AUC = {auc_xgb:.3f})", color='red', lw=2)
    
    nomo_roc_df = pd.read_csv(NOMOGRAM_ROC_DATA_PATH)
    nomo_combined = nomo_roc_df[nomo_roc_df['Cohort'] == 'Test_Combined']
    nomo_metrics_df = pd.read_csv(NOMOGRAM_METRICS_PATH)
    nomo_auc = nomo_metrics_df[nomo_metrics_df['Cohort'] == 'Test Set (Combined)']['AUC'].iloc[0]
    plt.plot(nomo_combined['FPR'], nomo_combined['TPR'], label=f"Nomogram (AUC = {nomo_auc:.3f})", linestyle='--', c='black', lw=2)
    
    plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC Curve (XGBoost vs. Nomogram) on {cohort_to_plot}'); plt.legend(); plt.grid(); 

    plt.show()
except Exception as e:
    print(f"  ROC 绘图失败: {e}")



print(f"\n--- 步骤9: 绘制 DCA 曲线 (XGBoost vs. Nomogram) ---")
plt.figure(figsize=(10, 8))
try:
    xgb_dca_df = dca_results_df[dca_results_df['model'] == 'XGBoost']

    nomo_dca_full_df = pd.read_csv(NOMOGRAM_DCA_CIC_DATA_PATH)
    if 'model' not in nomo_dca_full_df.columns and 'variable' in nomo_dca_full_df.columns:
        nomo_dca_full_df = nomo_dca_full_df.rename(columns={'variable': 'model'})
    

    nomo_dca_subset = nomo_dca_full_df[
        (nomo_dca_full_df['Cohort'] == "Combined Test Set") & 
        (nomo_dca_full_df['model'].isin(['Nomogram', 'all', 'none']))
    ]
    
    head_to_head_dca_df = pd.concat([xgb_dca_df, nomo_dca_subset])
    
    sns.lineplot(data=head_to_head_dca_df, x='threshold', y='net_benefit', hue='model', linewidth=2)
    plt.ylim(-0.1, 0.6)
    plt.title(f'DCA Curve (XGBoost vs. Nomogram) on {cohort_to_plot}')
    plt.xlabel('Threshold Probability'); plt.ylabel('Net Benefit'); plt.grid(True, linestyle='--', alpha=0.6); plt.legend(title='Models'); 
    
    plt.show()
    
except Exception as e:
    print(f"  DCA 绘图失败: {e}")


print(f"\n--- 步骤10: 绘制 CIC 曲线 (XGBoost vs. Nomogram) ---")
plt.figure(figsize=(12, 8))
try:
    xgb_cic_df = cic_df_long[cic_df_long['model'] == 'XGBoost']
    
    head_to_head_cic_df = pd.concat([xgb_cic_df, nomo_cic_long_df])
    head_to_head_cic_df['Curve_Type'] = head_to_head_cic_df['Curve_Type'].map({'n_high_risk': 'Number High Risk', 'n_high_risk_with_outcome': 'Number High Risk with Event'})

    sns.lineplot(data=head_to_head_cic_df, x='threshold', y='Number_of_Patients', hue='model', style='Curve_Type', linewidth=2)
    plt.title(f'CIC Curve (XGBoost vs. Nomogram) on {cohort_to_plot}'); plt.xlabel('High Risk Threshold'); plt.ylabel('Number of Patients'); plt.grid(True, linestyle='--', alpha=0.6); 
    plt.savefig("CIC_Curve_Head_to_Head.png")
    plt.show()
except Exception as e:
    print(f"  CIC 绘图失败: {e}")



print(f"\n--- 步骤11: DeLong's Test (在 {cohort_to_plot} 上) ---")
print(f"  最佳ML模型 (XGBoost) vs. R语言 Nomogram")

DELONG_NOMO_PREDS_PATH = "nomogram_raw_preds_for_delong.csv"

try:
    y_true_combined = all_datasets[cohort_to_plot][TARGET_VARIABLE]
    y_pred_xgb = all_dca_data[cohort_to_plot]['XGBoost']

    nomo_preds_df = pd.read_csv(DELONG_NOMO_PREDS_PATH)
    
    if len(y_true_combined) != len(nomo_preds_df):
        print(f"  DeLong 检验失败")


    y_pred_nomogram_R = nomo_preds_df['Nomogram_Pred_Probs']

    _, _, p_value = delong_roc_test(y_true_combined, y_pred_xgb, y_pred_nomogram_R)
    
    print(f"\n  DeLong's Test p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"  XGBoost 与 Nomogram 之间存在统计学上的显著差异。")
    else:
        print(f"  XGBoost 与 Nomogram 之间没有统计学上的显著差异。")
        
except FileNotFoundError:
    print(f"  找不到 Nomogram 原始预测文件: {DELONG_NOMO_PREDS_PATH}")
except Exception as e:
    print(f"  DeLong's Test 执行失败: {e}")
