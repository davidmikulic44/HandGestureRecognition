import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
NUM_CLASSES = 5

dataset = 'data.csv'
model_save_path = 'model'


x_dataset = np.loadtxt(dataset, 
                       delimiter=',', 
                       dtype='float32', 
                       usecols=list(range(1, (42) + 1)))
y_dataset = np.loadtxt(dataset, 
                       delimiter=',', 
                       dtype='int32', 
                       usecols=(0))

x_train, x_test, y_train, y_test = train_test_split(x_dataset, 
                                                    y_dataset, 
                                                    train_size=0.75, 
                                                    random_state=RANDOM_SEED,
                                                    stratify=y_dataset)



model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=128,
    validation_data=(x_test, y_test)
)   

model.summary()
train_loss = history.history['loss']
validation_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
model.save(model_save_path)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Prvi podgraf za točnost
plt.subplot(1, 2, 1)  # 1 red, 2 stupca, prvi graf
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()