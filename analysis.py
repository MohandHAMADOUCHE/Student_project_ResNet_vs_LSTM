import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np
from itertools import cycle
from tensorflow.keras.utils import to_categorical  # type: ignore
from PIL import Image

from training import load_and_compile_model, load_training_data
from utils import config, display_images, model_paths

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

# Plotar métricas comparativas entre modelos selecionados
def plot_metrics_comparison(reports, auc_scores, class_names):
    metrics = ['precision', 'recall', 'f1-score']
    
    for metric in metrics:
        plt.figure(figsize=(14, 6))
        for model_name, report in reports.items():
            scores = [report[class_name][metric] for class_name in class_names]
            plt.bar(np.arange(len(class_names)) + (list(reports.keys()).index(model_name) - 1) * 0.2, scores, width=0.2, label=model_name)
        plt.xlabel("Classes")
        plt.ylabel(metric.capitalize())
        plt.title(f"Comparison of {metric.capitalize()} across Models")
        plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
        plt.legend()
        save_figure_to_list()
    
    # Comparação do AUC
    plt.figure(figsize=(6, 6))
    for model_name, auc_score in auc_scores.items():
        plt.bar(model_name, auc_score)
    plt.title("Comparison of AUC across Models")
    plt.ylabel("AUC Score")
    save_figure_to_list()

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
    colors = cycle(plt.cm.tab10.colors)  # Usando cores genéricas do matplotlib
    for i, color in zip(range(len(class_names)), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title(f"ROC Curves - {method}")
    plt.legend(loc="lower right")
    save_figure_to_list()

# Plotar comparação de desempenho entre modelos
def plot_comparison(training_times, classification_times, histories):
    # Definir cores dinamicamente para a precisão (Training vs Validation)
    colors = cycle(plt.cm.tab10.colors)  # Usando cores genéricas do matplotlib
    
    # Plotar as comparações de precisão entre os modelos selecionados
    plt.figure(figsize=(14, 6))
    for model_name, history in histories.items():
        if model_name in config["selected_models"]:  # Verifica se o modelo foi selecionado
            epochs = range(1, len(history['accuracy']) + 1)
            plt.plot(epochs, history['accuracy'], label=f'{model_name} Training', color=next(colors))
            plt.plot(epochs, history['val_accuracy'], label=f'{model_name} Validation', linestyle='--', color=next(colors))
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    save_figure_to_list()
    
    # Comparar tempo de treinamento (apenas modelos selecionados)
    plt.figure(figsize=(8, 6))
    selected_training_times = {key: value for key, value in training_times.items() if key in config["selected_models"]}
    colors = cycle(plt.cm.tab10.colors)  # Usando um conjunto de cores do matplotlib
    plt.bar(selected_training_times.keys(), selected_training_times.values(), color=[next(colors) for _ in selected_training_times])
    plt.title('Training Time (seconds)')
    plt.ylabel('Time (seconds)')
    plt.legend()  # Garantindo que a legenda apareça para as barras
    save_figure_to_list()

    # Comparar tempo de classificação (apenas modelos selecionados)
    plt.figure(figsize=(8, 6))
    selected_classification_times = {key: value for key, value in classification_times.items() if key in config["selected_models"]}
    plt.bar(selected_classification_times.keys(), selected_classification_times.values(), color=[next(colors) for _ in selected_classification_times])
    plt.title('Classification Time (seconds)')
    plt.ylabel('Time (seconds)')
    plt.legend()  # Garantindo que a legenda apareça para as barras
    save_figure_to_list()


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

# Função principal para executar a análise
def do_all_analysis(X_test, y_test):
    y_true = y_test.argmax(axis=1)  # Garantir que y_true está em formato de rótulo (classe)
    reports = {}
    auc_scores = {}
    classification_times = {}

    plot_image_models("models.png")

    print("Loading pretrained models...")
    for model_name in config["selected_models"]:
        model = load_and_compile_model(model_paths[model_name])
        
        # Começar a classificação para cada modelo individualmente
        start_classification = time.time()
        y_pred = model.predict(X_test).argmax(axis=1)  # Obtenção da classe com maior probabilidade
        classification_times[model_name] = time.time() - start_classification
        
        # Calcular as métricas para cada modelo
        report, auc_score = evaluate_model_performance(y_true, y_pred, [f"Class {i}" for i in range(config["num_classes"])] )
        
        reports[model_name] = report
        auc_scores[model_name] = auc_score
        
        # Plotar matriz de confusão e curvas ROC individualmente para cada modelo
        plot_confusion_matrix(y_true, y_pred, [f"Class {i}" for i in range(config["num_classes"])], model_name)
        plot_multiclass_roc(y_true, y_pred, [f"Class {i}" for i in range(config["num_classes"])], model_name)
    
    print("Pretrained models loaded and compiled successfully!")

    # Plotar métricas comparativas entre modelos
    plot_metrics_comparison(reports, auc_scores, [f"Class {i}" for i in range(config["num_classes"])])
    
    # Plotar comparação de desempenho (tempo de treinamento, tempo de classificação)
    training_times, histories, _ = load_training_data()
    plot_comparison(training_times, classification_times, histories)
    
    display_images(image_list)
