from sklearn.metrics import classification_report, confusion_matrix
import logging


def compute_tpr_tnr_fpr_fnr(cm):
    """
    计算 TPR, TNR, FPR, FNR
    :param cm: 混淆矩阵 (confusion matrix)
    :return: TPR, TNR, FPR, FNR
    """
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[1, 0]
    FN = cm[0, 1]

    # 计算 TPR, TNR, FPR, FNR
    TPR = TP / (TP + FN)  # True Positive Rate
    TNR = TN / (TN + FP)  # True Negative Rate
    FPR = FP / (FP + TN)  # False Positive Rate
    FNR = FN / (FN + TP)  # False Negative Rate
    return TPR, TNR, FPR, FNR


def output_report(y, predicts):
    """
    评估模型性能，包括分类报告和 TPR、TNR 输出

    """

    # 使用classification_report输出详细的指标并格式化为四位小数
    report = classification_report(y, predicts.numpy(), target_names=["Normal", "Anomaly"],
                                         output_dict=True)

    # 格式化输出
    for label, metrics in report.items():
        if label != 'accuracy':  # 如果是 'accuracy'，跳过输出
            print(
                f"{label} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-score: {metrics['f1-score']:.4f}, Support: {metrics['support']:.4f}")

    # 计算TPR和TNR
    cm = confusion_matrix(y, predicts.numpy())
    TPR, TNR, FPR, FNR = compute_tpr_tnr_fpr_fnr(cm)
    print(f"TPR: {TPR:.4f}, TNR: {TNR:.4f}, FPR: {FPR:.4f}, FNR: {FNR:.4f}")

def out_log(y, predicts):
    """
    评估模型性能，包括分类报告和 TPR、TNR 输出
    """
    # 使用 classification_report 输出详细的指标，并格式化为四位小数
    report = classification_report(y, predicts.numpy(), target_names=["Normal", "Anomaly"],
                                  output_dict=True)

    # 格式化并写入日志
    log_output = []
    for label, metrics in report.items():
        if label != 'accuracy':  # 如果是 'accuracy'，跳过输出
            log_output.append(
                f"{label} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                f"F1-score: {metrics['f1-score']:.4f}, Support: {metrics['support']:.4f}")

    # 计算混淆矩阵
    cm = confusion_matrix(y, predicts.numpy())
    TPR, TNR, FPR, FNR = compute_tpr_tnr_fpr_fnr(cm)
    log_output.append(f"TPR: {TPR:.4f}, TNR: {TNR:.4f}, FPR: {FPR:.4f}, FNR: {FNR:.4f}")

    # 将所有结果写入日志
    logging.info("\n".join(log_output))  # 将所有结果连接为多行，写入日志文件
    logging.info("\n")
