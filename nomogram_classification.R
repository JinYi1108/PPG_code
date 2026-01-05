library(rms)
library(pROC)
library(caret)
library(precrec)
library(ResourceSelection)
library(dcurves)
library(ggplot2)
library(dplyr)


cat("-------------------------------------------------------\n")
cat("Nomogram：预测分类\n")
cat("-------------------------------------------------------\n\n")
cat("--- 步骤 1: 加载数据 ---\n")

train_file_path_class <- "111111.csv" 
test_a_data <- read.csv("a.csv")
test_b_data <- read.csv("b.csv")
test_c_data <- read.csv("c.csv")
combined_test_data <- rbind(test_a_data, test_b_data, test_c_data)

train_data <- read.csv(train_file_path_class)


cat(sprintf("训练集已加载: %d 行\n", nrow(train_data)))
cat(sprintf("测试集 A 已加载: %d 行\n", nrow(test_a_data)))
cat(sprintf("测试集 B 已加载: %d 行\n", nrow(test_b_data)))
cat(sprintf("测试集 C 已加载: %d 行\n", nrow(test_c_data)))
cat(sprintf("合并测试集已创建: %d 行\n", nrow(combined_test_data)))

cat("\n--- 步骤 2: 数据准备 ---\n")
predictor_names <- c("Hb", "PLT", "Portal.Vein.Thrombosis", "Ascites", "Esophageal.gastric.varices", "Splenomegaly")
categorical_features <- c("Portal.Vein.Thrombosis", "Ascites", "Esophageal.gastric.varices", "Splenomegaly")
target_binary_col_name <- "PPG" 

prepare_data <- function(df, train_levels_list) {
  for (col in categorical_features) {
    if (col %in% names(df)) {
 
      if (is.null(train_levels_list)) {
        df[[col]] <- factor(df[[col]])
      } else { 
        df[[col]] <- factor(df[[col]], levels = train_levels_list[[col]])
      }
    }
  }
  return(df)
}

train_data <- prepare_data(train_data, NULL)
train_factor_levels <- lapply(train_data[categorical_features], levels)


test_a_data <- prepare_data(test_a_data, train_factor_levels)
test_b_data <- prepare_data(test_b_data, train_factor_levels)
test_c_data <- prepare_data(test_c_data, train_factor_levels)
combined_test_data <- prepare_data(combined_test_data, train_factor_levels)
cat("所有数据集的分类变量已转换为因子。\n")


cat("\n--- 步骤 3: 训练 lrm 模型 ---\n")
dd_train <- datadist(train_data)
options(datadist = "dd_train")

set.seed(123)
model_formula <- as.formula(paste("PPG ~", paste(predictor_names, collapse = " + ")))
nomogram_model <- lrm(model_formula, data = train_data, x = TRUE, y = TRUE)
cat("模型训练完成。\n")






cat("\n--- 步骤 4.1: 创建并绘制列线图 ---\n")

nom_object_lrm <- nomogram(nomogram_model,
                                      fun = plogis,
                                     
    
                                      fun.at = c(0.01, 0.05, seq(0.2, 0.9, by = 0.2), 0.95, 0.99), 
                                      funlabel = paste("Probability of", target_binary_col_name, ">= 20 (i.e., Class 1)"),
                                      lp = FALSE 
)
plot(nom_object_lrm,
     xfrac = .30)


cat("列线图已绘制。\n")

cat("\n--- 步骤 4.2: 在所有数据集上生成预测概率 ---\n")
datasets <- list(
  Train = train_data,
  Test_A = test_a_data,
  Test_B = test_b_data,
  Test_C = test_c_data,
  Test_Combined = combined_test_data
)
predictions <- lapply(datasets, function(df) {
  predict(nomogram_model, newdata = df, type = "fitted")
})
cat("预测完成。\n")

cat("\n--- 步骤 5: 综合性能评估 ---\n")

cat("\n\n--- 5.1 完整的区分度评估 ---\n")

cat("\n--- a) 计算表观及外部验证的 ROC/AUC ---\n")

roc_list <- lapply(names(datasets), function(name) {
  if(length(unique(datasets[[name]]$PPG)) > 1) { 
    roc(datasets[[name]]$PPG, predictions[[name]], quiet = TRUE)
  }
})
names(roc_list) <- names(datasets)

roc_list <- roc_list[!sapply(roc_list, is.null)] 

cat("\n--- b) 执行 Bootstrap 内部验证 ---\n")
set.seed(123) 
val_results_boot <- validate(nomogram_model, method = "boot", B = 200) 

corrected_dxy <- val_results_boot["Dxy", "index.corrected"]
corrected_auc <- (corrected_dxy + 1) / 2

cat(sprintf("Bootstrap 校正后的 AUC: %.4f\n", corrected_auc))


cat("\n--- c) 创建综合性能总结表 ---\n")

auc_summary_external <- t(sapply(roc_list, function(r) {
  ci <- pROC::ci.auc(r)
  c(AUC = ci[2], LowerCI = ci[1], UpperCI = ci[3])
}))

summary_df <- data.frame(
  Validation_Type = c("Apparent Performance", "Internal Validation", "External Validation", "External Validation", "External Validation", "External Validation (Overall)"),
  Cohort = c("Training Set", "Training Set (Bootstrap Corrected)", "Test Set A", "Test Set B", "Test Set C", "Test Set (Combined)"),
  AUC = c(auc_summary_external["Train", "AUC"], corrected_auc, 
          auc_summary_external["Test_A", "AUC"], auc_summary_external["Test_B", "AUC"],
          auc_summary_external["Test_C", "AUC"], auc_summary_external["Test_Combined", "AUC"]),
  Lower_CI = c(auc_summary_external["Train", "LowerCI"], NA, 
               auc_summary_external["Test_A", "LowerCI"], auc_summary_external["Test_B", "LowerCI"],
               auc_summary_external["Test_C", "LowerCI"], auc_summary_external["Test_Combined", "LowerCI"]),
  Upper_CI = c(auc_summary_external["Train", "UpperCI"], NA,
               auc_summary_external["Test_A", "UpperCI"], auc_summary_external["Test_B", "UpperCI"],
               auc_summary_external["Test_C", "UpperCI"], auc_summary_external["Test_Combined", "UpperCI"])
)
numeric_columns <- c("AUC", "Lower_CI", "Upper_CI")
summary_df[numeric_columns] <- round(summary_df[numeric_columns], 3)


print(summary_df)



cat("\n--- d) 绘制对比 ROC 曲线图 ---\n")

plot(roc_list$Train, 
     main="ROC Curve Comparison", 
     col="black", 
     legacy.axes=TRUE,
     ylab="TPR",     
     xlab="FPR") 

plot(roc_list$Test_A, add=TRUE, col="blue")
plot(roc_list$Test_B, add=TRUE, col="red")
plot(roc_list$Test_C, add=TRUE, col="green")
plot(roc_list$Test_Combined, add=TRUE, col="purple", lwd=2)

legend_text <- sapply(names(roc_list), function(name) {
  sprintf("%s AUC = %.3f", name, pROC::auc(roc_list[[name]]))
})

legend("bottomright", legend=legend_text,
       col=c("black", "blue", "red", "green", "purple"), lwd=2,cex= 0.8)





cat("\n\n--- 5.2 完整的校准度评估 ---\n")



cat("\n--- a. 绘制校准曲线 ---\n")

set.seed(123)

cal_train_boot <- calibrate(nomogram_model, method = "boot", B = 200)
plot(cal_train_boot, 
     main = "Calibration Plot (Training Set, Corrected)",
     xlab = "Predicted Probability", 
     ylab = "Actual Probability")


par(mfrow = c(2, 2), pty = "s") 

for (name in c("Test_A", "Test_B", "Test_C", "Test_Combined")) {
  if (nrow(datasets[[name]]) >= 20 && length(unique(datasets[[name]]$PPG)) > 1) {
    

    val.prob(predictions[[name]], datasets[[name]]$PPG, pl = TRUE)
    

    title(main = paste("Calibration -", name),
          xlab = "Predicted Probability",
          ylab = "Actual Probability")
    
  } else {

    plot(0, type = 'n', axes = FALSE, xlab = "", ylab = "", main = paste("Calibration -", name))
    text(1, 0, "样本量过少或类别单一\n无法绘制校准曲线", cex = 1.2, pos=1)
  }
}
par(mfrow = c(1, 1), pty = "m") 


cat("\n\n--- b. 进行 Hosmer-Lemeshow 拟合优度检验 ---\n")

hl_results <- lapply(names(datasets), function(name) {
  if (length(unique(datasets[[name]]$PPG)) > 1) {
    

    true_outcomes <- datasets[[name]]$PPG
    predicted_probs <- predictions[[name]]
    
    complete_cases <- !is.na(predicted_probs)
    

    if (sum(!complete_cases) > 0) {
      cat(sprintf("  警告: 在数据集 '%s' 中发现并移除了 %d 个无法预测的样本(NA)。\n", name, sum(!complete_cases)))
    }
    

    true_outcomes_complete <- true_outcomes[complete_cases]
    predicted_probs_complete <- predicted_probs[complete_cases]
    

    if (length(predicted_probs_complete) < 20) {
      cat(sprintf("  注意: 在 '%s' 数据集移除NA后样本过少，跳过H-L检验。\n", name))
      return(NULL)
    }
    

    test <- hoslem.test(true_outcomes_complete, predicted_probs_complete, g = 10)

    
    return(c(X_squared = test$statistic, Df = test$parameter, p_value = test$p.value))
  }
})


hl_results <- hl_results[!sapply(hl_results, is.null)]
names(hl_results) <- names(datasets)[sapply(datasets, function(d) nrow(d) >= 20 && length(unique(d$PPG)) > 1)]


hl_summary <- do.call(rbind, hl_results)

print("Hosmer-Lemeshow Goodness of Fit Test Results:")
print(round(hl_summary, 3))



cat("\n\n--- 5.3 临床有效性评估 (DCA) ---\n")





dca_data <- bind_rows(
  mutate(datasets$Train, Cohort = "Train", Nomogram = predictions$Train),
  mutate(datasets$Test_A, Cohort = "Test_A", Nomogram = predictions$Test_A),
  mutate(datasets$Test_B, Cohort = "Test_B", Nomogram = predictions$Test_B),
  mutate(datasets$Test_C, Cohort = "Test_C", Nomogram = predictions$Test_C),
  mutate(datasets$Test_Combined, Cohort = "Test_Combined", Nomogram = predictions$Test_Combined)
) %>%
  mutate(PPG = as.integer(as.character(PPG)))



dca_datasets_list <- split(dca_data, dca_data$Cohort)
desired_order <- c("Train", "Test_A", "Test_B", "Test_C", "Test_Combined")
dca_datasets_list <- dca_datasets_list[desired_order]


dca_list <- lapply(dca_datasets_list, function(df) {
  if (length(unique(df$PPG)) > 1 && nrow(df) > 0) {
    dca(PPG ~ Nomogram,
        data = df,
        thresholds = seq(0.05, 0.95, by = 0.05))
  }
})
dca_list <- dca_list[!sapply(dca_list, is.null)]





dca_tidy_df <- bind_rows(
  lapply(dca_list, as_tibble), 
  .id = "Cohort"
)


dca_tidy_df$Cohort <- factor(dca_tidy_df$Cohort, levels = desired_order)


ggplot(data = dca_tidy_df, aes(x = threshold, y = net_benefit, color = label)) +
  geom_line(linewidth = 1.2) +
  facet_wrap(~ Cohort, ncol = 3) + 
  scale_y_continuous(limits = c(-0.1, 0.6), name = "Net Benefit") +
  scale_x_continuous(name = "Threshold Probability", limits = c(0, 1)) +
  labs(color = "Strategy") +
  theme_bw(base_size = 14) +
  ggtitle("Decision Curve Analysis by Cohort")


cat("\n决策曲线分析完成。\n")


cat("\n\n--- 5.4 综合分类指标总结 ---\n")





get_classification_metrics <- function(true_labels, pred_probs, threshold = 0.5) {

  if (length(unique(true_labels)) < 2) {

    return(setNames(rep(NA, 7), c("Accuracy", "Sensitivity", "Specificity", "Precision", "NPV", "F1_Score", "Balanced_Accuracy")))
  }
  

  predicted_class <- factor(ifelse(pred_probs >= threshold, 1, 0), levels = c(0, 1))
  true_labels_factor <- factor(true_labels, levels = c(0, 1))
  

  cm <- confusionMatrix(data = predicted_class, reference = true_labels_factor, positive = "1")
  

  accuracy <- cm$overall['Accuracy']
  sensitivity <- cm$byClass['Sensitivity'] 
  specificity <- cm$byClass['Specificity'] 
  precision <- cm$byClass['Precision']     
  npv <- cm$byClass['Neg Pred Value']     
  f1_score <- cm$byClass['F1']
  balanced_accuracy <- cm$byClass['Balanced Accuracy']
  

  return(c(Accuracy = accuracy, 
           Sensitivity = sensitivity, 
           Specificity = specificity,
           Precision = precision, 
           NPV = npv, 
           F1_Score = f1_score,
           Balanced_Accuracy = balanced_accuracy))
}


metrics_list <- lapply(names(datasets), function(name) {
  get_classification_metrics(datasets[[name]]$PPG, predictions[[name]])
})


metrics_summary_df <- do.call(rbind, metrics_list)
rownames(metrics_summary_df) <- names(datasets)


print(round(metrics_summary_df, 3))



get_metrics_with_ci <- function(true_labels, pred_probs, threshold = 0.5, R = 1000) {
  set.seed(123)
  

  calc_point_metrics <- function(labels, probs) {
    preds <- factor(ifelse(probs >= threshold, 1, 0), levels = c(0, 1))
    ref <- factor(labels, levels = c(0, 1))
    cm <- caret::confusionMatrix(preds, ref, positive = "1")
    
    acc <- cm$overall['Accuracy']
    sen <- cm$byClass['Sensitivity']
    pre <- cm$byClass['Precision']
    f1  <- cm$byClass['F1']
    return(c(acc, sen, pre, f1))
  }
  

  point_ests <- calc_point_metrics(true_labels, pred_probs)
  

  boot_results <- replicate(R, {
    idx <- sample(1:length(true_labels), replace = TRUE)
    if(length(unique(true_labels[idx])) < 2) return(rep(NA, 4))
    calc_point_metrics(true_labels[idx], pred_probs[idx])
  })
  
  ci_lower <- apply(boot_results, 1, function(x) quantile(x, 0.025, na.rm = TRUE))
  ci_upper <- apply(boot_results, 1, function(x) quantile(x, 0.975, na.rm = TRUE))
  
  res_formatted <- sapply(1:4, function(i) {
    sprintf("%.3f (%.3f-%.3f)", point_ests[i], ci_lower[i], ci_upper[i])
  })
  
  names(res_formatted) <- c("Accuracy", "Recall", "Precision", "F1_Score")
  return(res_formatted)
}


cat("\n\n--- 95% CI 的综合性能指标 ---\n")

performance_list <- list()

for (name in names(datasets)) {
  df <- datasets[[name]]
  preds <- predictions[[name]]
  
  if (nrow(df) > 0 && length(unique(df$PPG)) > 1) {
    roc_obj <- pROC::roc(df$PPG, preds, quiet = TRUE)
    auc_ci <- pROC::ci.auc(roc_obj)
    auc_str <- sprintf("%.3f (%.3f-%.3f)", auc_ci[2], auc_ci[1], auc_ci[3])

    class_metrics <- get_metrics_with_ci(df$PPG, preds, threshold = 0.5, R = 1000)
    
    performance_list[[name]] <- data.frame(
      Cohort = name,
      AUC_95CI = auc_str,
      Accuracy_95CI = class_metrics["Accuracy"],
      Recall_95CI = class_metrics["Recall"],
      Precision_95CI = class_metrics["Precision"],
      F1_95CI = class_metrics["F1_Score"],
      stringsAsFactors = FALSE
    )
  }
}


final_summary_with_ci <- do.call(rbind, performance_list)

val_results_boot <- validate(nomogram_model, method = "boot", B = 200)
corrected_auc <- (val_results_boot["Dxy", "index.corrected"] + 1) / 2
final_summary_with_ci <- rbind(
  final_summary_with_ci,
  data.frame(Cohort = "Train_Bootstrap_Corrected", 
             AUC_95CI = sprintf("%.3f (N/A)", corrected_auc), 
             Accuracy_95CI="N/A", Recall_95CI="N/A", Precision_95CI="N/A", F1_95CI="N/A")
)

print(final_summary_with_ci)

write.csv(final_summary_with_ci, "nomogram_performance_with_95CI.csv", row.names = FALSE)
cat("\n已将带 95% CI 的性能指标保存到 'nomogram_performance_with_95CI.csv'\n")














cat("\n\n--- 步骤 6: 保存ROC曲线坐标数据 ---\n")


roc_data_list <- lapply(names(roc_list), function(name) {
  roc_obj <- roc_list[[name]]
  

  roc_df <- data.frame(
    TPR = roc_obj$sensitivities,
    FPR = 1 - roc_obj$specificities,
    Thresholds = roc_obj$thresholds,
    Cohort = name 
  )
  return(roc_df)
})


roc_curves_df <- do.call(rbind, roc_data_list)


write.csv(roc_curves_df, "nomogram_roc_curve_data.csv", row.names = FALSE)
cat("\n已将所有ROC曲线的坐标数据保存到 'nomogram_roc_curve_data.csv'\n")

print("\nROC坐标数据预览 (前5行):")
head(roc_curves_df)


cat("\n\n--- 步骤 7: 保存综合性能指标 ---\n")


performance_list <- list()

for (name in names(datasets)) {
  
  df <- datasets[[name]]
  preds <- predictions[[name]]

  auc_val <- NA; lower_ci <- NA; upper_ci <- NA
  brier_val <- NA; hl_p_val <- NA
  accuracy <- NA; sensitivity <- NA; specificity <- NA
  precision <- NA; npv <- NA; f1_score <- NA
  

  if (nrow(df) > 0 && length(unique(df$PPG)) > 1) {

    roc_obj <- pROC::roc(df$PPG, preds, quiet = TRUE)
    auc_ci_vector <- pROC::ci.auc(roc_obj)
    auc_val <- auc_ci_vector[2]
    lower_ci <- auc_ci_vector[1]
    upper_ci <- auc_ci_vector[3]
    

    brier_val <- mean((preds - df$PPG)^2, na.rm = TRUE)
    

    complete_cases_preds <- !is.na(preds)
    if(sum(complete_cases_preds) >= 20) {
      hl_test <- hoslem.test(df$PPG[complete_cases_preds], preds[complete_cases_preds], g = 10)
      hl_p_val <- hl_test$p.value
    }
    

    predicted_class <- factor(ifelse(preds >= 0.5, 1, 0), levels = c(0, 1))
    true_labels_factor <- factor(df$PPG, levels = c(0, 1))
    
   
    if (all(levels(true_labels_factor) %in% unique(true_labels_factor)) && 
        all(levels(predicted_class) %in% unique(predicted_class))) {
      
      cm <- caret::confusionMatrix(data = predicted_class, reference = true_labels_factor, positive = "1")
      accuracy <- cm$overall['Accuracy']
      sensitivity <- cm$byClass['Sensitivity']
      specificity <- cm$byClass['Specificity']
      precision <- cm$byClass['Precision']    
      npv <- cm$byClass['Neg Pred Value']
      f1_score <- cm$byClass['F1']
    }
  }
  

  performance_list[[name]] <- data.frame(
    Cohort_ID = name, AUC = auc_val, Lower_CI = lower_ci, Upper_CI = upper_ci,
    Brier_Score = brier_val, HL_p_value = hl_p_val, Accuracy = accuracy,
    Sensitivity = sensitivity, Specificity = specificity, Precision = precision,
    NPV = npv, F1_Score = f1_score
  )
}


summary_base <- do.call(rbind, performance_list)

val_results_boot <- validate(nomogram_model, method = "boot", B = 200)
corrected_dxy <- val_results_boot["Dxy", "index.corrected"]
corrected_auc <- (corrected_dxy + 1) / 2

final_performance_summary <- data.frame(
  Validation_Type = c("Apparent Performance", "Internal Validation", "External Validation", 
                      "External Validation", "External Validation", "External Validation (Overall)"),
  Cohort = c("Training Set", "Training Set (Bootstrap Corrected)", "Test Set A", 
             "Test Set B", "Test Set C", "Test Set (Combined)"),
  stringsAsFactors = FALSE
)

data_rows <- summary_base[c("Train", "Test_A", "Test_B", "Test_C", "Test_Combined"), ]


final_performance_summary <- final_performance_summary %>%
  left_join(
    bind_rows(
     
      data.frame(Cohort="Training Set", 
                 AUC=data_rows["Train", "AUC"], Lower_CI=data_rows["Train", "Lower_CI"], Upper_CI=data_rows["Train", "Upper_CI"], 
                 Brier_Score=data_rows["Train", "Brier_Score"], HL_p_value=data_rows["Train", "HL_p_value"], 
                 Accuracy=data_rows["Train", "Accuracy"], 
                 Recall=data_rows["Train", "Sensitivity"],
                 Specificity=data_rows["Train", "Specificity"], 
                 Precision=data_rows["Train", "Precision"], 
                 NPV=data_rows["Train", "NPV"], 
                 `F1-Score`=data_rows["Train", "F1_Score"]), 
      
      data.frame(Cohort="Training Set (Bootstrap Corrected)", AUC=corrected_auc, Lower_CI=NA, Upper_CI=NA, Brier_Score=NA, HL_p_value=NA, Accuracy=NA, Recall=NA, Specificity=NA, Precision=NA, NPV=NA, `F1-Score`=NA),
      
      data.frame(Cohort="Test Set A", 
                 AUC=data_rows["Test_A", "AUC"], Lower_CI=data_rows["Test_A", "Lower_CI"], Upper_CI=data_rows["Test_A", "Upper_CI"], 
                 Brier_Score=data_rows["Test_A", "Brier_Score"], HL_p_value=data_rows["Test_A", "HL_p_value"], 
                 Accuracy=data_rows["Test_A", "Accuracy"], 
                 Recall=data_rows["Test_A", "Sensitivity"],
                 Specificity=data_rows["Test_A", "Specificity"], 
                 Precision=data_rows["Test_A", "Precision"], 
                 NPV=data_rows["Test_A", "NPV"], 
                 `F1-Score`=data_rows["Test_A", "F1_Score"]), 
      
      data.frame(Cohort="Test Set B", 
                 AUC=data_rows["Test_B", "AUC"], Lower_CI=data_rows["Test_B", "Lower_CI"], Upper_CI=data_rows["Test_B", "Upper_CI"], 
                 Brier_Score=data_rows["Test_B", "Brier_Score"], HL_p_value=data_rows["Test_B", "HL_p_value"], 
                 Accuracy=data_rows["Test_B", "Accuracy"], 
                 Recall=data_rows["Test_B", "Sensitivity"], 
                 Specificity=data_rows["Test_B", "Specificity"], 
                 Precision=data_rows["Test_B", "Precision"], 
                 NPV=data_rows["Test_B", "NPV"], 
                 `F1-Score`=data_rows["Test_B", "F1_Score"]), 
      
      data.frame(Cohort="Test Set C", 
                 AUC=data_rows["Test_C", "AUC"], Lower_CI=data_rows["Test_C", "Lower_CI"], Upper_CI=data_rows["Test_C", "Upper_CI"], 
                 Brier_Score=data_rows["Test_C", "Brier_Score"], HL_p_value=data_rows["Test_C", "HL_p_value"], 
                 Accuracy=data_rows["Test_C", "Accuracy"], 
                 Recall=data_rows["Test_C", "Sensitivity"], 
                 Specificity=data_rows["Test_C", "Specificity"], 
                 Precision=data_rows["Test_C", "Precision"], 
                 NPV=data_rows["Test_C", "NPV"], 
                 `F1-Score`=data_rows["Test_C", "F1_Score"]), 
      
      data.frame(Cohort="Test Set (Combined)", 
                 AUC=data_rows["Test_Combined", "AUC"], Lower_CI=data_rows["Test_Combined", "Lower_CI"], Upper_CI=data_rows["Test_Combined", "Upper_CI"], 
                 Brier_Score=data_rows["Test_Combined", "Brier_Score"], HL_p_value=data_rows["Test_Combined", "HL_p_value"], 
                 Accuracy=data_rows["Test_Combined", "Accuracy"], 
                 Recall=data_rows["Test_Combined", "Sensitivity"], 
                 Specificity=data_rows["Test_Combined", "Specificity"], 
                 Precision=data_rows["Test_Combined", "Precision"], 
                 NPV=data_rows["Test_Combined", "NPV"], 
                 `F1-Score`=data_rows["Test_Combined", "F1_Score"]) 
    ),
    by = "Cohort"
  )


print("最终性能指标汇总表:")

numeric_cols <- names(final_performance_summary)[sapply(final_performance_summary, is.numeric)]

final_performance_summary[numeric_cols] <- round(final_performance_summary[numeric_cols], 3)
print(final_performance_summary)



write.csv(final_performance_summary, "nomogram_performance_metrics_full.csv", row.names = FALSE)
cat("\n已将完整的综合性能指标保存到 'nomogram_performance_metrics_full.csv'\n")



cat("\n\n--- 步骤 8: 计算并保存DCA和CIC数据 ---\n")



all_datasets_list <- list(
  "Train Set" = train_data,
  "Test Set A" = test_a_data,
  "Test Set B" = test_b_data,
  "Test Set C" = test_c_data,
  "Combined Test Set" = combined_test_data
)


all_dca_results_list <- list()
all_cic_results_list <- list()

thresholds_seq <- seq(0.01, 0.99, by = 0.01)

for (cohort_name in names(all_datasets_list)) {
  
  current_data <- all_datasets_list[[cohort_name]]
  
  if (nrow(current_data) > 0 && length(unique(current_data$PPG)) > 1) {
    

    pred_probs <- predict(nomogram_model, newdata = current_data, type = "fitted")
    
    dca_temp_df <- data.frame(
      PPG = current_data[[target_binary_col_name]],
      Nomogram = pred_probs
    )
    
 
    dca_result_obj <- dca(
      data = dca_temp_df,
      formula = PPG ~ Nomogram,
      thresholds = thresholds_seq
    )
  
    dca_result_tibble <- as_tibble(dca_result_obj, type = "decision_curve")
 
    all_dca_results_list[[cohort_name]] <- dca_result_tibble

    cic_results <- sapply(thresholds_seq, function(thresh) {
      high_risk_mask <- pred_probs >= thresh
      n_risk <- sum(high_risk_mask, na.rm = TRUE)
      n_risk_pos <- sum(high_risk_mask & (current_data$PPG == 1), na.rm = TRUE)
      return(c(n_risk = n_risk, n_risk_pos = n_risk_pos))
    })
    
    cic_df <- data.frame(
      threshold = thresholds_seq,
      variable = "Nomogram", 
      label = "Nomogram", 
      n_risk = cic_results["n_risk", ],
      n_risk_pos = cic_results["n_risk_pos", ]
    )
  
    all_cic_results_list[[cohort_name]] <- cic_df
    
  } else {
    cat(sprintf("跳过数据集 '%s' (样本量过少或类别单一)\n", cohort_name))
  }
}


final_dca_data <- bind_rows(all_dca_results_list, .id = "Cohort")
output_dca_file <- "nomogram_DCA_data.csv"
write.csv(final_dca_data, output_dca_file, row.names = FALSE)


final_cic_data <- bind_rows(all_cic_results_list, .id = "Cohort")
output_cic_file <- "nomogram_CIC_data.csv"
write.csv(final_cic_data, output_cic_file, row.names = FALSE)


cat(sprintf("DCA数据 (包含 'All', 'None' 和 'Nomogram' 的 net_benefit) 已保存到: %s\n", output_dca_file))
cat(sprintf("CIC数据 (仅包含 'Nomogram' 的 n_risk, n_risk_pos) 已保存到: %s\n", output_cic_file))



cat("\n\n--- 步骤 9: 保存 DeLong 检验所需的原始预测数据 ---\n")

y_true_nomogram <- datasets$Test_Combined$PPG

y_pred_nomogram <- predictions$Test_Combined

delong_data <- data.frame(
  PPG_True = y_true_nomogram,
  Nomogram_Pred_Probs = y_pred_nomogram
)

output_delong_file <- "nomogram_raw_preds_for_delong.csv"
write.csv(delong_data, output_delong_file, row.names = FALSE)

cat(sprintf("DeLong 检验所需的 Nomogram 预测数据已保存到: %s\n", output_delong_file))

