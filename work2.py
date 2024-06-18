import pandas as pd

# 載入數據
train_df = pd.read_csv('assignment-2-hand-gesture-recognition/train_data.csv')
test_df = pd.read_csv('assignment-2-hand-gesture-recognition/test_data.csv')

"""# 檢查數據結構
print(train_df.head())
print(test_df.head())
"""

import numpy as np

# 轉換訓練數據
X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('float32')
y_train = train_df.iloc[:, 0].values

# 轉換測試數據
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)
X_test = scaler.transform(X_test.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# 第一個卷積層
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 第二個卷積層
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 將特徵圖展平
model.add(Flatten())

# 全連接層
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# 輸出層
model.add(Dense(25, activation='softmax'))  # 24個類別

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

"""
#輸出成CSV檔案
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
samp_ids = range(1, len(predicted_classes) + 1)
results_df = pd.DataFrame({'samp_id': samp_ids, 'label': predicted_classes})
results_df.to_csv('prediction_results.csv', index=False)
"""