import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings

warnings.filterwarnings("ignore")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score
from scipy.stats import chi2
import xgboost as xgb


CONFIG_FILE_PATH = 'best_model_configs.json'
TRAIN_FILE_PATH = '1111.xlsx'
TEST_A_PATH = 'a.xlsx'
TEST_B_PATH = 'b.xlsx'
TEST_C_PATH = 'c.xlsx'

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

def evaluate_subgroup(model, subgroup_df, features):

    if subgroup_df.empty:
        print("亚组为空，无法评估。")
        return None
        
    X_test = subgroup_df[features]
    y_true = subgroup_df[TARGET_VARIABLE]
    
    if len(np.unique(y_true)) < 2:
        print(f"亚组只包含一个类别 (n={len(y_true)})。跳过评估。")
        return None

    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = (y_pred_proba >= 0.5)

    
    metrics = {}
    metrics['N_Patients'] = len(y_true)
    metrics['N_Positive'] = int(y_true.sum())
    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
    _, metrics['HL_p_value'] = hosmer_lemeshow_test(y_true, y_pred_proba)
    metrics['Accuracy'] = accuracy_score(y_true, y_pred_class)
    metrics['Recall'] = recall_score(y_true, y_pred_class)
    metrics['Precision'] = precision_score(y_true, y_pred_class)
    metrics['F1-Score'] = f1_score(y_true, y_pred_class)
    

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    metrics['roc_data'] = {'fpr': fpr, 'tpr': tpr}
    
    return metrics

def main():
    print("--- 亚组分析 ---")
    
    try:
        
        print("\n--- 步骤1: 重新训练 XGBoost ---")
        
       
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)['XGBoost']
        features, params = config['features'], config['params']
        print(f"  将使用 {len(features)} 个特征: {features}")

    
        train_df = pd.read_excel(TRAIN_FILE_PATH)
        X_train = train_df[features]
        y_train = train_df[TARGET_VARIABLE]
 
        counts = y_train.value_counts()
        scale_pos_weight_value = counts[0] / counts[1]
        
     
        base_model = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss', scale_pos_weight=scale_pos_weight_value)
        base_model.set_params(**params)
        
        calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
        calibrated_model.fit(X_train, y_train)
        
        print("  XGBoost 模型训练并校准完毕。")

    
        print("\n--- 步骤2: 正在加载并拆分 Combined Test Set ---")
        
        df_A = pd.read_excel(TEST_A_PATH)
        df_B = pd.read_excel(TEST_B_PATH)
        df_C = pd.read_excel(TEST_C_PATH)
        df_combined = pd.concat([df_A, df_B, df_C], ignore_index=True)
        
    
        if 'Aetiology' not in df_combined.columns:
            print(f"'Aetiology' 列在测试集文件中不存在。无法进行亚组分析。")
            return

    
        sub_non_sinu = df_combined[df_combined['Aetiology'].isin([0, 1, 3])].copy()
        sub_sinu = df_combined[df_combined['Aetiology'] == 2].copy()
        
        print(f"  - 非窦性 (Aetiology 0,1,3): {len(sub_non_sinu)} 名患者")
        print(f"  - 窦性 (Aetiology 2):     {len(sub_sinu)} 名患者")

    
        print("\n--- 步骤3: 评估两个亚组 ---")
        
        metrics_non_sinu = evaluate_subgroup(calibrated_model, sub_non_sinu, features)
        metrics_sinu = evaluate_subgroup(calibrated_model, sub_sinu, features)
        
        

    
        print("\n--- 步骤4: 绘制亚组ROC对比图 ---")
        
        plt.figure(figsize=(10, 8))
        
        if metrics_non_sinu:
            roc_data = metrics_non_sinu['roc_data']
            auc = metrics_non_sinu['AUC']
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                     label=f"Non-Sinu (0,1,3) (AUC = {auc:.3f}, n={metrics_non_sinu['N_Patients']})", 
                     lw=2)
            
        if metrics_sinu:
            roc_data = metrics_sinu['roc_data']
            auc = metrics_sinu['AUC']
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                     label=f"Sinu (2) (AUC = {auc:.3f}, n={metrics_sinu['N_Patients']})", 
                     lw=2, linestyle='--')

        plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'XGBoost Subgroup ROC Analysis (Aetiology) on Combined Test Set')
        plt.legend()
        plt.grid(True)
    
        plt.show()


    
        print("\n--- 步骤5: 亚组性能对比表 (XGBoost) ---")
        
    
        if metrics_non_sinu and metrics_sinu:
            table_data = {
                "Non-Sinu (0,1,3)": metrics_non_sinu,
                "Sinu (2)": metrics_sinu
            }
        
            table_data["Non-Sinu (0,1,3)"].pop('roc_data')
            table_data["Sinu (2)"].pop('roc_data')
            
            df_table = pd.DataFrame(table_data)
            print(df_table.round(4))
        else:
            print("  未能成功评估两个亚组，无法生成表格。")
        
    except FileNotFoundError as e:
        print(f"\n找不到文件 {e.filename}。")
    except Exception as e:
        print(f"\n--- 发生意外错误 ---")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()