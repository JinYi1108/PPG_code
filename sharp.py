import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import shap 
import xgboost as xgb 
import joblib
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False  


CONFIG_FILE_PATH = 'best_model_configs.json'
TRAIN_FILE_PATH = '1111.xlsx'
TEST_A_PATH = 'a.xlsx'
TEST_B_PATH = 'b.xlsx'
TEST_C_PATH = 'c.xlsx'

TARGET_VARIABLE = 'PPG'
RANDOM_STATE = 42

def main():
    print("--- SHAP 分析 ---")
    
    try:
        
        print("\n--- 步骤1: 训练XGBoost 模型 ---")
        
        
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)['XGBoost']
        features, params = config['features'], config['params']
        print(f"  将使用 {len(features)} 个特征: {features}")

        
        train_df = pd.read_excel(TRAIN_FILE_PATH)
        X_train = train_df[features]
        y_train = train_df[TARGET_VARIABLE]
        
      
        counts = y_train.value_counts()
        scale_pos_weight_value = counts[0] / counts[1]
        
       
        base_xgboost_model = xgb.XGBClassifier(
            random_state=RANDOM_STATE, 
            n_jobs=-1, 
            eval_metric='logloss', 
            scale_pos_weight=scale_pos_weight_value
        )
        base_xgboost_model.set_params(**params)
        base_xgboost_model.fit(X_train, y_train)
        
        print("  XGBoost 模型训练完成")
        joblib.dump(base_xgboost_model, 'xgb_base_model_for_shap.pkl')
        print("  -> 'xgb_base_model_for_shap.pkl' 已保存")
       
        print("\n--- 步骤2: 加载 Combined Test Set  ---")
        df_A = pd.read_excel(TEST_A_PATH)
        df_B = pd.read_excel(TEST_B_PATH)
        df_C = pd.read_excel(TEST_C_PATH)
        df_combined = pd.concat([df_A, df_B, df_C], ignore_index=True)
        X_test = df_combined[features] 
        
        print(f"  测试集加载，维度: {X_test.shape}")

        print("\n--- 步骤3: 计算 SHAP 值 ---")
        

        explainer = shap.TreeExplainer(base_xgboost_model)
        shap_explanation = explainer(X_test)
        vals = getattr(shap_explanation, "values", None)


        if vals is None:
            shap_values_raw = explainer.shap_values(X_test)
            if isinstance(shap_values_raw, list) and len(shap_values_raw) >= 2:
                shap_matrix = np.array(shap_values_raw[1]) 
            else:
                shap_matrix = np.array(shap_values_raw)
        else:
            vals = np.array(vals)
            if vals.ndim == 3:
        
                shap_matrix = vals[:, :, 1]
            elif vals.ndim == 2:
                shap_matrix = vals
            else:

                if vals.ndim == 1:

                    shap_matrix = vals.reshape(1, -1)
                else:
                    raise ValueError(f"Unexpected shap values shape: {vals.shape}")


        if shap_matrix.ndim != 2:
            raise ValueError(f"shap_matrix must be 2D (n_samples,n_features), got shape {shap_matrix.shape}")



        print("  SHAP 值计算完毕")
        
  
        print("\n--- 步骤4: 绘制 SHAP 全局解释图 ---")
        

        plt.figure()
        shap.summary_plot(shap_matrix, X_test, show=False) 
        #plt.title('SHAP Summary Plot (Beeswarm) for Positive Class (PPG>=20)')
        plt.tight_layout()
        #plt.savefig("SHAP_Summary_Beeswarm.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("  蜂群图 (Beeswarm) 已保存为: SHAP_Summary_Beeswarm.png")

        plt.figure()
        shap.summary_plot(shap_matrix, X_test, plot_type="bar", show=False)
        #plt.title('SHAP Feature Importance (Mean Absolute SHAP Value)')
        plt.tight_layout()
        #plt.savefig("SHAP_Summary_Bar.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("  条形图 (Bar) 已保存为: SHAP_Summary_Bar.png")


        

        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
        feature_names = X_test.columns


        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_matrix, X_test, show=False)

        ax = plt.gca()


        yticks = ax.get_yticks()
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        xmin, xmax = ax.get_xlim()


        feature_order = [label for label in ylabels]
        mean_abs_shap_ordered = pd.Series(mean_abs_shap, index=feature_names).loc[feature_order]
        bar_scale_factor = 3.0 

        scaled_bar_widths = mean_abs_shap_ordered.values * bar_scale_factor


        bars = ax.barh(
            y=yticks,
            width=scaled_bar_widths,
            
            left=xmin,
            color='lightgray',
            alpha=0.5,   
            height=0.7,  
            align='center',
           
            zorder=0
        )
        total_mean_abs_shap = mean_abs_shap_ordered.sum()
        max_scaled_bar_right_edge = xmin + scaled_bar_widths.max()

        max_scaled_bar_right_edge = xmin + scaled_bar_widths.max()
        content_xmax = max(xmax, max_scaled_bar_right_edge) 


        plot_right_padding = (content_xmax - xmin) * 0.05 
        final_xmax = content_xmax + plot_right_padding
        

        text_align_x = final_xmax - (final_xmax - xmin) * 0.001
        

        for i, bar in enumerate(bars):
            y_pos = bar.get_y() + bar.get_height() / 2 
            
            original_bar_width = mean_abs_shap_ordered.iloc[i] 
            percentage = (original_bar_width / total_mean_abs_shap) * 100
            label_text = f"{original_bar_width:.3f} ({percentage:.1f}%)" 
            

            text_x_pos = text_align_x
            

            ax.text(
                x=text_x_pos, 
                y=y_pos, 
                s=label_text, 
                va='center', 
                ha='right',   
                fontsize=7   
            )
        


        # text_left_offset = (xmax - xmin) * 0.01
        # for i, bar in enumerate(bars):
        #     y_pos = bar.get_y() + bar.get_height() / 2 
            

        #     original_bar_width = mean_abs_shap_ordered.iloc[i] 
            

        #     percentage = (original_bar_width / total_mean_abs_shap) * 100
            

        #     label_text = f"{original_bar_width:.3f} ({percentage:.1f}%)" 
            

        #     text_x_pos = xmin + text_left_offset 
            

        #     ax.text(
        #         x=text_x_pos, 
        #         y=y_pos, 
        #         s=label_text, 
        #         va='center', 
        #         ha='left',   
        #         fontsize=9  
        #     )


        
            


        ax.legend(loc='lower right', frameon=False)
        ax.set_xlabel("SHAP value (impact on model output)")
        plt.title("Mean(|SHAP|value)", fontsize=13)
        plt.tight_layout()
        plt.savefig("SHAP_Summary_Overlay.png", dpi=300, bbox_inches='tight')
        plt.show()









        patient_index = 41
        shap_instance = shap_explanation[patient_index]  

        if isinstance(explainer.expected_value, (list, np.ndarray)):
            if len(explainer.expected_value) > 1:
                base_value_class1 = explainer.expected_value[1]
            else:
                base_value_class1 = explainer.expected_value[0]
        else:
            base_value_class1 = explainer.expected_value
        #shap.initjs()


        try:
           
            force_plot_html = shap.force_plot(
                base_value_class1,                      
                shap_instance.values,
                X_test.iloc[patient_index, :],   
                show=False                                  
            )

            if force_plot_html is not None:
                save_path = f"SHAP_Force_Plot_Patient_{patient_index}.html"
                shap.save_html(save_path, force_plot_html)
                print(f"  力图 (Force Plot HTML) 已保存为: {save_path}")

            else:
                print("  未能生成 Force Plot HTML 对象。")

        except Exception as e:
            print(f"  为病人 {patient_index} 生成 Force Plot HTML 时出错: {e}")


        

    except FileNotFoundError as e:
        print(f"\n找不到文件 {e.filename}")
    except ImportError:
        print("找不到 shap 库")
    except Exception as e:
        print(f"\n--- 发生意外错误 ---")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()