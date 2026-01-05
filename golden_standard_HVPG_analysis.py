import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings

warnings.filterwarnings("ignore")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.stats import chi2
import xgboost as xgb


CONFIG_FILE_PATH = 'best_model_configs.json'
TRAIN_FILE_PATH = '11111.xlsx'

TIPS_TEST_FILE_PATH ='xxxxx.xlsx'

TARGET_VARIABLE = 'PPG' 
RANDOM_STATE = 42

def main():
    print("--- 步骤1: 训练 XGBoost---")
    
    try:
        
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
        
        print("  XGBoost 模型训练并校准完毕")

    except FileNotFoundError as e:
        print(f"\n找不到文件 {e.filename}。")
        return
    except KeyError:
        print("\n 'best_model_configs.json' 中没有 'XGBoost' ")
        return
    except Exception as e:
        print(f"\n模型训练失败: {e}")
        return

    
    print(f"\n--- 步骤2: 加载TIPS验证集 '{TIPS_TEST_FILE_PATH}' ---")
    
    try:
        df_tips = pd.read_excel(TIPS_TEST_FILE_PATH)
        
        required_cols = [TARGET_VARIABLE, 'HVPG'] + features
        missing_cols = [col for col in required_cols if col not in df_tips.columns]
        
        if missing_cols:
            print(f" '{TIPS_TEST_FILE_PATH}' 文件中缺少: {missing_cols}")
            return
            
        print(f"  成功加载TIPS验证集。样本量 n = {len(df_tips)}")


        
        X_test_tips = df_tips[features]
        y_true_tips = df_tips[TARGET_VARIABLE] 
        
        
        y_score_xgb = calibrated_model.predict_proba(X_test_tips)[:, 1]
        print(y_score_xgb)
        
        y_score_gold = df_tips['HVPG'] + 1
        y_score_gold_binary = (y_score_gold >= 20).astype(int)
        print(y_score_gold)

        
        if len(np.unique(y_true_tips)) < 2:
            print("IPS验证集只包含一个类别 无法绘制ROC/PR曲线")
            return

    except FileNotFoundError:
        print(f"找不到TIPS验证文件 '{TIPS_TEST_FILE_PATH}'")
        return
    except Exception as e:
        print(f"\n 加载TIPS数据或预测时失败: {e}")
        return


    print("\n--- 步骤3: 生成 ROC 曲线对比图 ---")
    try:
        fpr_xgb, tpr_xgb, _ = roc_curve(y_true_tips, y_score_xgb)
        auc_xgb = roc_auc_score(y_true_tips, y_score_xgb)
        
        fpr_gold, tpr_gold, _ = roc_curve(y_true_tips, y_score_gold_binary)
        auc_gold = roc_auc_score(y_true_tips, y_score_gold_binary)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost Model (AUC = {auc_xgb:.3f})", color='red', lw=2.5)
        plt.plot(fpr_gold, tpr_gold, label=f"Gold Standard (HVPG+1) (AUC = {auc_gold:.3f})", color='blue', linestyle='--', lw=2.5)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC Curve Comparison on TIPS Subgroup (n={len(df_tips)})')
        plt.legend(loc='lower right')
        plt.grid(True)

        plt.show()
        print(f"  ROC对比图已保存为: ROC_Curve_TIPS_Showdown.png")

    except Exception as e:
        print(f"  ROC 绘图失败: {e}")


    print("\n--- 步骤4: 生成 Precision-Recall (PR) 曲线对比图 ---")
    try:
        prec_xgb, recall_xgb, _ = precision_recall_curve(y_true_tips, y_score_xgb)
        ap_xgb = average_precision_score(y_true_tips, y_score_xgb)
        
        prec_gold, recall_gold, _ = precision_recall_curve(y_true_tips, y_score_gold)
        ap_gold = average_precision_score(y_true_tips, y_score_gold)

        plt.figure(figsize=(10, 8))
        
        plt.plot(recall_xgb, prec_xgb, label=f"XGBoost Model (AP = {ap_xgb:.3f})", color='red', lw=2.5)
        plt.plot(recall_gold, prec_gold, label=f"Gold Standard (HVPG+1) (AP = {ap_gold:.3f})", color='blue', linestyle='--', lw=2.5)
        
        
        no_skill = y_true_tips.mean()
        plt.plot([0, 1], [no_skill, no_skill], 'k--', label=f'No-Skill (AP = {no_skill:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve on TIPS Subgroup (n={len(df_tips)})')
        plt.legend(loc='upper right')
        plt.grid(True)
        #plt.savefig("PR_Curve_TIPS_Showdown.png")
        plt.show()
        print(f"  PR对比图已保存为: PR_Curve_TIPS_Showdown.png")

    except Exception as e:
        print(f"  PR 绘图失败: {e}")

if __name__ == "__main__":
    main()
