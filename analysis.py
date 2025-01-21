import json
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np
from itertools import cycle
from tensorflow.keras.utils import to_categorical  # type: ignore
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import shutil
import pickle
from training import load_and_compile_model, load_training_data

# Salvar uma figura na lista de imagens
def save_figure_to_list(image_list):
    import io
    from PIL import Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_list.append(np.array(image))
    buf.close()
    plt.close()

# Display all generated plots interactively
def display_images(image_list):
    if not image_list:
        print("No images were generated.")
        return

    current_image_index = 0

    def update_image():
        plt.clf()
        plt.imshow(image_list[current_image_index])
        plt.axis('off')
        plt.title(f"Image {current_image_index + 1} of {len(image_list)}")
        plt.draw()

    def on_key(event):
        nonlocal current_image_index
        if event.key == 'right':
            current_image_index = (current_image_index + 1) % len(image_list)
        elif event.key == 'left':
            current_image_index = (current_image_index - 1) % len(image_list)
        update_image()

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)
    update_image()
    plt.show()

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
def plot_metrics_comparison(reports, auc_scores, class_names, image_list):
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
        save_figure_to_list(image_list)
    
    # Comparação do AUC
    plt.figure(figsize=(6, 6))
    for model_name, auc_score in auc_scores.items():
        plt.bar(model_name, auc_score)
    plt.title("Comparison of AUC across Models")
    plt.ylabel("AUC Score")
    save_figure_to_list(image_list)

# Plotar matriz de confusão
def plot_confusion_matrix(y_true, y_pred, class_names, method, image_list):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {method}")
    save_figure_to_list(image_list)

# Plotar curvas ROC para classificação multiclasse
def plot_multiclass_roc(y_true, y_pred, class_names, method, image_list):
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
    save_figure_to_list(image_list)

# Plotar comparação de desempenho entre modelos
def plot_comparison(training_times, classification_times, histories, image_list, config):
    """
    Compara desempenho de modelos selecionados com base em métricas de treino, validação, 
    tempo de treinamento e classificação.
    """
    # Comparar precisão (accuracy) entre os modelos
    plt.figure(figsize=(14, 6))
    colors = cycle(plt.cm.tab10.colors)  # Usar ciclo de cores
    for model_name, is_selected in config["selected_models"].items():
        if not is_selected:
            continue
        
        # Obter métricas do modelo
        model_history = histories.get(model_name, {})
        training_accuracy = model_history.get('accuracy', [])
        validation_accuracy = model_history.get('val_accuracy', [])
        epochs = range(1, len(training_accuracy) + 1)

        # Plotar treino e validação
        color = next(colors)
        plt.plot(epochs, training_accuracy, label=f'{model_name} Training', color=color)
        plt.plot(epochs, validation_accuracy, label=f'{model_name} Validation', linestyle='--', color=color)
    
    # Configuração do gráfico
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    save_figure_to_list(image_list)

    # Comparar tempo de treinamento (apenas modelos selecionados)
    plt.figure(figsize=(8, 6))
    selected_training_times = {key: value for key, value in training_times.items() if config["selected_models"].get(key, False)}
    colors = cycle(plt.cm.tab10.colors)
    plt.bar(selected_training_times.keys(), selected_training_times.values(), color=[next(colors) for _ in selected_training_times])
    plt.title('Training Time (seconds)')
    plt.ylabel('Time (seconds)')
    save_figure_to_list(image_list)

    # Comparar tempo de classificação (apenas modelos selecionados)
    plt.figure(figsize=(8, 6))
    selected_classification_times = {key: value for key, value in classification_times.items() if config["selected_models"].get(key, False)}
    colors = cycle(plt.cm.tab10.colors)
    plt.bar(selected_classification_times.keys(), selected_classification_times.values(), color=[next(colors) for _ in selected_classification_times])
    plt.title('Classification Time (seconds)')
    plt.ylabel('Time (seconds)')
    save_figure_to_list(image_list)

# Função principal para executar a análise
def do_all_analysis(X_test, y_test, config, class_to_index, image_list):
    y_true = y_test.argmax(axis=1) if len(y_test.shape) > 1 else y_test
    reports, auc_scores, classification_times = {}, {}, {}

    for model_name, is_selected in config["selected_models"].items():
        if not is_selected:
            continue
        print(f"Loading model: {model_name}")
        model = load_and_compile_model(f"{model_name}_model.keras")
        # Começar a classificação para cada modelo individualmente
        start_classification = time.time()
        y_pred = model.predict(X_test).argmax(axis=1)
        classification_times[model_name] = time.time() - start_classification
    
        class_names = [f"Class {i}" for i in range(len(class_to_index))]
        report, auc_score = evaluate_model_performance(y_true, y_pred, class_names)
        reports[model_name] = report
        auc_scores[model_name] = auc_score
        
        plot_confusion_matrix(y_true, y_pred, class_names, model_name, image_list)
        plot_multiclass_roc(y_true, y_pred, class_names, model_name, image_list)

    # Plotar métricas comparativas entre modelos
    plot_metrics_comparison(reports, auc_scores, [f"Class {i}" for i in range(len(class_to_index))], image_list)
    
    # Plotar comparação de desempenho (tempo de treinamento, tempo de classificação)
    training_times, histories, _ = load_training_data()
    plot_comparison(training_times, classification_times, histories, image_list, config)
    
    #save_analysis_results(reports, training_times, classification_times, auc_scores, image_list, model_paths, config)
    
    print("Analysis complete. Displaying results...")
    display_images(image_list)


# Função para criar o diretório
def create_model_directory(base_dir="Models"):
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    path = os.path.join(base_dir, timestamp)
    os.makedirs(path, exist_ok=True)
    return path

# Função para copiar os modelos para o diretório da análise
def copy_models_to_directory(model_paths, target_directory):
    models_dir = os.path.join(target_directory, "models")
    os.makedirs(models_dir, exist_ok=True)
    for model_name, src_path in model_paths.items():
        dest_path = os.path.join(models_dir, f"{model_name}.keras")
        shutil.copy(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")

# Função para gerar o relatório em PDF
def generate_pdf_report(directory, model_reports, training_times, classification_times, auc_scores, images, config):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Adicionar página inicial com título
    pdf.add_page()
    pdf.set_font("Arial", size=14, style="B")
    pdf.cell(200, 10, txt="Model Performance Report", ln=True, align="C")
    pdf.ln(10)

    # Adicionar dados de treinamento e classificação
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Training and Classification Times", ln=True, align="L")
    for model, time in training_times.items():
        pdf.cell(200, 8, txt=f"Model: {model} | Training Time: {time:.2f}s | Classification Time: {classification_times.get(model, 0):.2f}s", ln=True)
    pdf.ln(10)

    # Adicionar pontuação AUC
    pdf.cell(200, 10, txt="AUC Scores", ln=True, align="L")
    for model, auc in auc_scores.items():
        pdf.cell(200, 8, txt=f"Model: {model} | AUC Score: {auc:.2f}", ln=True)
    pdf.ln(10)

    # Adicionar relatórios detalhados de cada modelo
    for model, report in model_reports.items():
        pdf.add_page()
        pdf.set_font("Arial", size=12, style="B")
        pdf.cell(200, 10, txt=f"Model: {model}", ln=True)
        pdf.set_font("Arial", size=10)
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):  # Classes individuais
                pdf.cell(200, 8, txt=f"Class: {class_name}", ln=True)
                for metric, value in metrics.items():
                    pdf.cell(200, 8, txt=f"  {metric}: {value:.2f}", ln=True)
            else:  # Métricas gerais como 'accuracy'
                pdf.cell(200, 8, txt=f"{class_name}: {metrics:.2f}", ln=True)
        pdf.ln(5)

    # Adicionar imagens geradas
    for idx, img_array in enumerate(images):
        pdf.add_page()
        pdf.set_font("Arial", size=12, style="B")
        pdf.cell(200, 10, txt=f"Figure {idx + 1}", ln=True)
        image_path = os.path.join(directory, f"figure_{idx + 1}.png")
        Image.fromarray(img_array).save(image_path)
        pdf.image(image_path, x=10, y=30, w=180)
        pdf.ln(85)  # Espaço para evitar sobreposição

    # Adicionar configurações no PDF
    pdf.add_page()
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(200, 10, txt="Configuration Settings", ln=True)
    pdf.set_font("Arial", size=10)
    for key, value in config.items():
        if isinstance(value, dict):
            pdf.cell(200, 8, txt=f"{key}: ", ln=True)
            for sub_key, sub_value in value.items():
                pdf.cell(200, 8, txt=f"  {sub_key}: {sub_value}", ln=True)
        else:
            pdf.cell(200, 8, txt=f"{key}: {value}", ln=True)
    pdf.ln(5)

    # Salvar o PDF
    pdf_path = os.path.join(directory, "model_report.pdf")
    pdf.output(pdf_path)
    print(f"PDF report saved at: {pdf_path}")

# Função principal para salvar os dados e relatório
def save_analysis_results(model_reports, training_times, classification_times, auc_scores, images, model_paths, config, training_data_file="training_times.json"):
    directory = create_model_directory()

    # Copiar modelos para o diretório
    copy_models_to_directory(model_paths, directory)

    # Salvar arquivo training_times.json no diretório
    with open(os.path.join(directory, "training_times.json"), "w") as json_file:
        json.dump(training_times, json_file, indent=4)
    
    # Salvar os relatórios de modelos no subdiretório
    models_dir = os.path.join(directory, "models")
    for model_name, model in model_reports.items():
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    # Gerar e salvar o relatório em PDF
    generate_pdf_report(directory, model_reports, training_times, classification_times, auc_scores, images, config)

    print(f"Results saved in directory: {directory}")