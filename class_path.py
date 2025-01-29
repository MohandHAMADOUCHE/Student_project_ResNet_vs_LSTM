import os
import shutil
import pandas as pd

# Caminho do CSV e da pasta raiz do dataset
csv_path = r"C:\Users\Gustavo\Desktop\UrbanSound8K\metadata\UrbanSound8K.csv"
audio_root_path = r"C:\Users\Gustavo\Desktop\UrbanSound8K\audio"

# Carregar o arquivo CSV
df = pd.read_csv(csv_path)

# Processar cada linha no CSV
for _, row in df.iterrows():
    file_name = row["slice_file_name"]
    fold = row["fold"]
    
    # Caminho original e destino do arquivo
    original_path = os.path.join(audio_root_path, file_name)
    destination_folder = os.path.join(audio_root_path, f"fold{fold}")
    destination_path = os.path.join(destination_folder, file_name)

    # Criar a pasta destino se não existir
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Mover o arquivo para a pasta correspondente
    if os.path.exists(original_path):
        shutil.move(original_path, destination_path)
        print(f"Movido: {original_path} -> {destination_path}")
    else:
        print(f"Arquivo não encontrado: {original_path}")

print("Organização concluída!")
