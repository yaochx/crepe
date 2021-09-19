import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('TensorFlow:', tf.__version__)

for cap in ['tiny', 'small', 'medium', 'large', 'full']:
    capacity_multiplier = {
        'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
    }[cap]
    layers = [1, 2, 3, 4, 5, 6]
    filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
    widths = [512, 64, 64, 64, 64, 64]
    strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    x = Input(shape=(1024,), name='input', dtype='float32')
    y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

    for l, f, w, s in zip(layers, filters, widths, strides):
        y = Conv2D(f, (w, 1), strides=s, padding='same',
                    activation='relu', name="conv%d" % l)(y)
        y = BatchNormalization(name="conv%d-BN" % l)(y)
        y = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid',
                        name="conv%d-maxpool" % l)(y)
        y = Dropout(0.25, name="conv%d-dropout" % l)(y)

    y = Permute((2, 1, 3), name="transpose")(y)
    y = Flatten(name="flatten")(y)
    y = Dense(360, activation='sigmoid', name="classifier")(y)

    model = Model(inputs=x, outputs=y)

    print('model-'+cap+'.h5')
    model.load_weights('model-'+cap+'.h5')

    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    graph_info = profile(forward_pass.get_concrete_function().graph,
                            options=ProfileOptionBuilder.float_operation())

    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    flops = graph_info.total_float_ops // 2
    print('Flops: {:,}'.format(flops))

print(36829876/1024/1024)
print(113505460/1024/1024)
print(386394292/1024/1024)
print(818666676/1024/1024)
print(1410322612/1024/1024)