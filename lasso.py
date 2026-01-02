import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False  



df_full = pd.read_excel('xxxx.xlsx')



X = df_full.drop('PPG', axis=1)
y = df_full['PPG']
feature_names = X.columns.tolist()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
print(f"数据准备完成。共 {X_scaled_df.shape[1]} 个特征进行LASSO筛选。")

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
Cs_grid = np.logspace(-3, 2, 50)
lasso_cv = LogisticRegressionCV(
    Cs=Cs_grid,
    cv=cv_strategy,
    penalty='l1',
    solver='liblinear',
    scoring='roc_auc',
    random_state=42,
    max_iter=2000,
    n_jobs=-1
)
lasso_cv.fit(X_scaled_df, y)
print(f"\n交叉验证完成。")

scores = lasso_cv.scores_[1]
mean_scores = np.mean(scores, axis=0)
std_error = np.std(scores, axis=0) / np.sqrt(cv_strategy.n_splits)

best_C_index = np.argmax(mean_scores)
best_C = Cs_grid[best_C_index]
best_score = mean_scores[best_C_index]
best_se = std_error[best_C_index]

one_se_threshold = best_score - best_se
one_se_C_index = np.where(mean_scores >= one_se_threshold)[0][0]
one_se_C = Cs_grid[one_se_C_index]

print(f"  - 性能最佳的C值 (对应λ_min): {best_C:.4f} (AUC = {best_score:.3f})")
print(f"  - '一倍标准误'规则选择的C值 (对应λ_1se): {one_se_C:.4f} (AUC = {mean_scores[one_se_C_index]:.3f})")

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 7))
plt.plot(np.log10(Cs_grid), mean_scores, color='red', marker='o', markersize=5, label='Mean AUC')
plt.fill_between(np.log10(Cs_grid), mean_scores - std_error, mean_scores + std_error, alpha=0.15, color='red')
plt.axvline(np.log10(best_C), linestyle='--', color='black', label=f'C_min_error (λ_min)')
plt.axvline(np.log10(one_se_C), linestyle=':', color='blue', label=f'C_1se (λ_1se)')
plt.title('LASSO Cross-Validation (Binomial Deviance equivalent with AUC)', fontsize=16)
plt.xlabel('Log10(C) [Regularization: Strong -> Weak]', fontsize=12)
plt.ylabel('Mean AUC Score', fontsize=12)
plt.legend(loc='lower right')
plt.savefig('LASSO交叉验证曲线.png', dpi=300)
plt.show()

coefs_path = []
for c in Cs_grid:
    model = LogisticRegression(penalty='l1', C=c, solver='liblinear', random_state=42, max_iter=2000)
    model.fit(X_scaled_df, y)
    coefs_path.append(model.coef_.flatten())
coefs_path = np.array(coefs_path)


plt.figure(figsize=(12, 7))


for i in range(coefs_path.shape[1]):
    plt.plot(np.log10(Cs_grid), coefs_path[:, i], label=feature_names[i])


plt.axvline(np.log10(best_C), linestyle='--', color='black', label=f'C_min_error (λ_min)')
plt.axvline(np.log10(one_se_C), linestyle=':', color='blue', label=f'C_1se (λ_1se)')

plt.title('LASSO Coefficient Path', fontsize=16)
plt.xlabel('Log10(C) [Regularization: Strong -> Weak]', fontsize=12)
plt.ylabel('Coefficients', fontsize=12)


plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('LASSO系数路径图.png', dpi=300)
plt.show()


model_1se = LogisticRegression(penalty='l1', C=one_se_C, solver='liblinear', random_state=42, max_iter=2000)
model_1se.fit(X_scaled_df, y)
coefs_1se = model_1se.coef_.flatten()
features_1se = [feature for feature, coef in zip(feature_names, coefs_1se) if coef != 0]

print(f"\n--- 筛选结果对比 ---")
print(f"\n1. 根据 '一倍标准误' 规则 (C={one_se_C:.4f})，选定的特征 (最简模型):")
if features_1se:
    print(f"   共 {len(features_1se)} 个特征被选中:")
    for feature in features_1se:
        print(f"     - {feature}")
else:
    print("   没有特征被选中。")


model_min_error = LogisticRegression(penalty='l1', C=best_C, solver='liblinear', random_state=42, max_iter=2000)
model_min_error.fit(X_scaled_df, y)
coefs_min_error = model_min_error.coef_.flatten()
features_min_error = [feature for feature, coef in zip(feature_names, coefs_min_error) if coef != 0]

print(f"\n2. 根据 '最小误差' 规则 (C={best_C:.4f})，选定的特征 (最佳性能模型):")
if features_min_error:
    print(f"   共 {len(features_min_error)} 个特征被选中:")
    for feature in features_min_error:
        print(f"     - {feature}")
else:
    print("   没有特征被选中。")







from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df_features):


    vif_data = pd.DataFrame()
    vif_data["feature"] = df_features.columns
    

    vif_data["VIF"] = [variance_inflation_factor(df_features.values, i) 
                       for i in range(df_features.shape[1])]
    vif_data["Tolerance"] = 1 / vif_data["VIF"]
    

    return vif_data.sort_values(by="VIF", ascending=False)



print("--- 1. 对20个特征进行VIF检验 ---")
vif_for_ml = calculate_vif(X_scaled_df)
print(vif_for_ml)

features_for_nomogram = [
    'Hb', 'PLT', 'Portal Vein Thrombosis', 'Ascites', 
    'Esophageal gastric varices', 'Splenomegaly'
]
X_scaled_nomogram = X_scaled_df[features_for_nomogram]

print("\n\n--- 2. 对最终用于Nomogram的6个特征进行VIF检验 ---")
vif_for_nomogram = calculate_vif(X_scaled_nomogram)
print(vif_for_nomogram)


if vif_for_nomogram['VIF'].max() < 5:
    print("\n用于Nomogram的特征集不存在显著的多重共线性问题 (所有VIF < 5)。")
elif vif_for_nomogram['VIF'].max() < 10:
    print("\n用于Nomogram的特征集存在轻微的多重共线性问题,但在可接受范围内 (所有VIF < 10)。")
else:
    print("\n用于Nomogram的特征集存在严重的多重共线性问题 (有VIF >= 10),建议进一步处理。")
