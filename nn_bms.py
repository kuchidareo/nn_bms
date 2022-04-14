from tabnanny import verbose
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

## xslx -> tf_dataset
df = pd.read_excel("data.xlsx", header = 0)
del df["BMS_ID"]

target = df.pop("BMS")

# normalize
df_ = df.copy()
for col in df_.columns.values:
    s = df_[col]
    df[col] = (s - s.mean()) / s.std()
del df_

# split train to test
train_d, test_d = train_test_split(df, test_size = 0.2)
train_t, test_t = train_test_split(target, test_size = 0.2)

dataset_train = tf.data.Dataset.from_tensor_slices((train_d.values, train_t.values))
dataset_test = tf.data.Dataset.from_tensor_slices((test_d.values, test_t.values))

## machine leaning by using nn

train_dataset = dataset_train.shuffle(len(train_d)).batch(1)
test_dataset = dataset_test.shuffle((len(test_d))).batch(1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

# predictions = model(train_dataset)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_dataset, epochs = 15)

print(model.evaluate(test_dataset, verbose = 2))







