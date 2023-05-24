import tensorflow as tf


window_size = 20
batch_size =32
shuffle_buffer_size = 1000
x_train = []

# first we have to divide our data into features or the x_labels and labels
# the feature will be the number of the values in the series with our label being the next value
# we take a window of the dataset we have and train the model using the

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    #first it generates a tf dataset from time series values.
    dataset = tf.data.Dataset.from_tensor_slices(series)

    #will take the dataset,with a specific windowsize, drops others

    dataset = dataset.window(window_size + 1,shift=1, drop_remainder=True)

    #flattens the windowset by putting its element in a signle  batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # creates tuples with features and labells
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window :(window[:-1],window[-1]))

    #shuffles windows
    
    dataset  = dataset.batch(batch_size).prefetch(1)
    return dataset


dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, input_shape = [window_size], activation='relu'),
    tf.keras.layers.Dense(10, actiivation='relu'),
    tf.keras.layers.Dense(1),

])
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6,momentum=0.9))

model.fit(dataset,epochs=100,verbose=0)