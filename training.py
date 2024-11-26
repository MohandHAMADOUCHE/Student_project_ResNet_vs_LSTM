from tensorflow.keras import layers, models # type: ignore

# Build an LSTM model for classification
def build_lstm_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))  # Adiciona explicitamente a camada Input
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Build a ResNet model for classification
def build_resnet_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    for _ in range(3):
        shortcut = x
        x = layers.Conv1D(64, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(64, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = layers.Conv1D(64, kernel_size=1, padding='same')(shortcut)
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)

# Build a Transformer model for classification
def build_transformer_model(input_shape, num_classes, num_heads=4, ff_dim=128, num_layers=2):
    inputs = layers.Input(shape=input_shape)

    # Embed the input
    x = layers.Dense(ff_dim, activation='relu')(inputs)
    
    for _ in range(num_layers):
        # Multi-head attention block
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
        
        # Feed-forward block
        ffn_output = layers.Dense(ff_dim, activation='relu')(attention_output)
        ffn_output = layers.Dense(ff_dim)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
    
    # Global pooling and classification layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)
