#  from tensorflow.keras.models import load_model
import tensorflow as tf
from keras.models import load_model
import argparse
import sys, os

sys.path.append(os.path.join(sys.path[0], './modules/'))
import my_losses

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='set input arguments')
parser.add_argument('-inputm', action="store",
                                    dest='inputm', type=str, default='model.h5')
args = parser.parse_args()

class CustomTrainStep(tf.keras.Model):
    def __init__(self, n_gradients=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        #  config['n_gradients'] = self.n_gradients
        config['n_gradients'] = 4
        return config

#tf.keras.backend.clear_session()
model = load_model(args.inputm, 
                           #  custom_objects={'tf':tf}) # add any other custom object you might need ("e.g. a masked loss")
                           custom_objects={'masked_loss':my_losses.masked_loss,
                                           'multitask_loss': my_losses.multitask_loss,
                                           'masked_loss_binary': my_losses.masked_loss_binary,
                                           'masked_loss_categorical': my_losses.masked_loss_categorical,
                                           'CustomTrainStep': CustomTrainStep})
'''
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_weights.h5")
'''
model.save("saved_model")