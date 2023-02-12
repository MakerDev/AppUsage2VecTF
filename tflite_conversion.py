import tensorflow as tf
import os
import utils
from AppUsage2VecDataset import AppUsage2VecDataset
from AppUsage2VecTFLite import AppUsage2VecTFLite
from torch.utils.data import DataLoader

os.environ['MLIR_CRASH_REPRODUCER_DIRECTORY'] = 'enable'

args      = utils.parse_args()
num_users = len(open(os.path.join('data', 'user2id.txt'), 'r').readlines())
num_apps  = len(open(os.path.join('data', 'app2id.txt'), 'r').readlines())

latest_model_dir = tf.train.latest_checkpoint('checkpoints/epoch8')
model: AppUsage2VecTFLite = tf.saved_model.load('checkpoints/test')

# data load
mini = True
dataset = AppUsage2VecDataset(mode='test', mini=mini)
dataloader = iter(DataLoader(dataset, batch_size=1))
data, targets = next(dataloader)

users     = tf.convert_to_tensor(data[0])
time_vecs = tf.convert_to_tensor(data[1])
app_seqs  = tf.convert_to_tensor(data[2])
time_seqs = tf.convert_to_tensor(data[3])

model.predict_sample(app_seqs, time_seqs, users, time_vecs)

# Convert the model
# converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
converter = tf.lite.TFLiteConverter.from_saved_model('checkpoints/test')  # path to the SavedModel directory
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
signatures = interpreter.get_signature_list()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(interpreter.get_input_details())
print(interpreter.get_output_details())

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
