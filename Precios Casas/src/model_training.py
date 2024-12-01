import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Paso 1: Cargar datos
df = pd.read_csv('data/train.csv')

# Paso 2: Selección de características y objetivo
X = df[['LotArea', 'OverallQual', 'YearBuilt']]  # Características
y = df['SalePrice']  # Variable objetivo

# Paso 3: Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Paso 5: Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Una salida para predicción de precios
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Paso 6: Entrenar el modelo
model.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_test_scaled, y_test))

# Paso 7: Guardar el modelo
model.save('models/house_price_model.h5')
print("Modelo guardado en 'models/house_price_model.h5'")
