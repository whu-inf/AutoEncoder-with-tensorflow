import numpy as np
import h5py
import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import model_selection
from Model_define_tf import Encoder, Decoder, NMSE

# parameters
feedback_bits = 128
img_height = 16
img_width = 32
img_channels = 2
tf.random.set_seed(323)

# Model construction
# encoder model
Encoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="encoder_input")
Encoder_output = Encoder(Encoder_input, feedback_bits)
encoder = keras.Model(inputs=Encoder_input, outputs=Encoder_output, name='encoder')

# decoder model
Decoder_input = keras.Input(shape=(feedback_bits,), name='decoder_input')
Decoder_output = Decoder(Decoder_input, feedback_bits)
decoder = keras.Model(inputs=Decoder_input, outputs=Decoder_output, name="decoder")

# autoencoder model
autoencoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="original_img")
encoder_out = encoder(autoencoder_input)
decoder_out = decoder(encoder_out)
autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

# Data loading
data_load_address = './drive/My Drive'
mat = h5py.File(data_load_address+'/H_train.mat', 'r')
data = np.transpose(mat['H_train'])      # shape=(320000, 1024)
data = data.astype('float32')
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
data = np.transpose(data, (0, 2, 3, 1))   # change to data_form: 'channel_last'
x_train, x_test = sklearn.model_selection.train_test_split(data, test_size=0.05, random_state=1)

class CustomLearningRateScheduler(keras.callbacks.Callback):

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))

#lr scheduler
def lr_scheduler(epoch, lr):
  if epoch < 20:
    return 1e-3 * (1+(epoch/10))
  else:
    return 1.5e-3 *(1+tf.math.cos((epoch-20)*pi/80))

class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('on_epoch_end() called')
    y_test = autoencoder.predict(x_test, batch_size=512)
    print('The NMSE is ' + np.str(NMSE(x_test, y_test)))
    

# model training
autoencoder.fit(x=x_train, y=x_train, batch_size=256, epochs=50, callbacks=[CustomCallback(),
        CustomLearningRateScheduler(lr_scheduler),], verbose=1, validation_split=0.1)

# model save
# save encoder
modelsave1 = './Modelsave/encoder.h5'
encoder.save(modelsave1)
# save decoder
modelsave2 = './Modelsave/decoder.h5'
decoder.save(modelsave2)

# model test
y_test = autoencoder.predict(x_test, batch_size=512)
print('The NMSE is ' + np.str(NMSE(x_test, y_test)))
