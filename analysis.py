import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np
from itertools import cycle
from tensorflow.keras.utils import to_categorical # type: ignore
from PIL import Image

# Lista para armazenar imagens geradas
image_list = []

# Salvar uma figura na lista de imagens
def save_figure_to_list():
    import io
    from PIL import Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_list.append(np.array(image))
    buf.close()
    plt.close()

# Avaliar desempenho do modelo
def evaluate_model_performance(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    y_true_binary = to_categorical(y_true, num_classes=len(class_names))
    y_pred_binary = to_categorical(y_pred, num_classes=len(class_names))
    try:
        auc_score = roc_auc_score(y_true_binary, y_pred_binary, average='macro', multi_class='ovr')
    except ValueError:
        auc_score = None
    return report, auc_score

def plot_metrics_comparison(report_resnet, report_lstm, report_transformer, auc_resnet, auc_lstm, auc_transformer, class_names):
    """
    Plots comparison of precision, recall, f1-score, and AUC between ResNet, LSTM, and Transformer models.
    """
    metrics = ['precision', 'recall', 'f1-score']

    for metric in metrics:
        resnet_scores = [report_resnet[class_name][metric] for class_name in class_names]
        lstm_scores = [report_lstm[class_name][metric] for class_name in class_names]
        transformer_scores = [report_transformer[class_name][metric] for class_name in class_names]

        x = np.arange(len(class_names))
        width = 0.25

        plt.figure(figsize=(14, 6))
        plt.bar(x - width, resnet_scores, width, label='ResNet')
        plt.bar(x, lstm_scores, width, label='LSTM')
        plt.bar(x + width, transformer_scores, width, label='Transformer')
        plt.xlabel("Classes")
        plt.ylabel(metric.capitalize())
        plt.title(f"Comparison of {metric.capitalize()} across Models")
        plt.xticks(ticks=x, labels=class_names, rotation=45)
        plt.legend()
        save_figure_to_list()
        plt.close()

    # Comparação do AUC
    plt.figure(figsize=(6, 6))
    plt.bar(['ResNet', 'LSTM', 'Transformer'], [auc_resnet, auc_lstm, auc_transformer], color=['blue', 'orange', 'green'])
    plt.title("Comparison of AUC across Models")
    plt.ylabel("AUC Score")
    save_figure_to_list()  # Salva o gráfico na lista de imagens
    plt.close()  # Fecha o gráfico para evitar sobrecarga

# Plotar matriz de confusão
def plot_confusion_matrix(y_true, y_pred, class_names, method):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {method}")
    save_figure_to_list()

# Plotar curvas ROC para classificação multiclasse
def plot_multiclass_roc(y_true, y_pred, class_names, method):
    y_true_bin = to_categorical(y_true, num_classes=len(class_names))
    y_pred_bin = to_categorical(y_pred, num_classes=len(class_names))
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(len(class_names)), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title(f"ROC Curves - {method}")
    plt.legend(loc="lower right")
    save_figure_to_list()

def plot_comparison(
    resnet_train_time, lstm_train_time, transformer_train_time, 
    resnet_class_time, lstm_class_time, transformer_class_time,
    history_resnet, history_lstm, history_transformer
):
    # Acessando os históricos diretamente como dicionários
    epochs_resnet = range(1, len(history_resnet.history['accuracy']) + 1)
    epochs_lstm = range(1, len(history_lstm.history['accuracy']) + 1)
    epochs_transformer = range(1, len(history_transformer.history['accuracy']) + 1)

    plt.plot(epochs_resnet, history_resnet.history['accuracy'], label='ResNet Training')
    plt.plot(epochs_resnet, history_resnet.history['val_accuracy'], label='ResNet Validation', linestyle='--')
    plt.plot(epochs_lstm, history_lstm.history['accuracy'], label='LSTM Training')
    plt.plot(epochs_lstm, history_lstm.history['val_accuracy'], label='LSTM Validation', linestyle='--')
    plt.plot(epochs_transformer, history_transformer.history['accuracy'], label='Transformer Training')
    plt.plot(epochs_transformer, history_transformer.history['val_accuracy'], label='Transformer Validation', linestyle='--')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    save_figure_to_list()  # Salva a imagem do gráfico
    plt.close()  # Fecha o gráfico

    # Comparar tempo de treinamento
    plt.figure(figsize=(8, 6))
    plt.bar(['ResNet', 'LSTM', 'Transformer'], [resnet_train_time, lstm_train_time, transformer_train_time], color=['blue', 'orange', 'green'])
    plt.title('Training Time (seconds)')
    plt.ylabel('Time (seconds)')
    save_figure_to_list()  # Salva a imagem do gráfico
    plt.close()  # Fecha o gráfico

    # Comparar tempo de classificação
    plt.figure(figsize=(8, 6))
    plt.bar(['ResNet', 'LSTM', 'Transformer'], [resnet_class_time, lstm_class_time, transformer_class_time], color=['blue', 'orange', 'green'])
    plt.title('Classification Time (seconds)')
    plt.ylabel('Time (seconds)')
    save_figure_to_list()  # Salva a imagem do gráfico
    plt.close()  # Fecha o gráfico


def plot_image_models(image_path):
    """
    Exibe uma imagem no plot do matplotlib.

    :param image_path: Caminho para a imagem no computador.
    :param method: Nome ou descrição do método, usado como título do gráfico.
    """
    # Carregar a imagem
    img = Image.open(image_path)

    # Configurar o plot
    plt.figure(figsize=(20, 16))
    plt.imshow(img)
    plt.axis("off")  # Remove os eixos para focar na imagem
    
    save_figure_to_list()  # Salva a imagem do gráfico
    plt.close()  # Fecha o gráfico

