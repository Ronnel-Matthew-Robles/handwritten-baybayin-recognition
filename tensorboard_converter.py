import numpy as np
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = 'logs'
csv_dir = 'csv'

if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

for root, dirs, files in os.walk(log_dir):
    for d in dirs:
        model_dir = os.path.join(root, d)
        print(f'Processing model: {model_dir}')
        for subdir in os.listdir(model_dir):
            if subdir == 'train' or subdir == 'validation':
                event_file = os.path.join(model_dir, subdir, os.listdir(os.path.join(model_dir, subdir))[0])
                event_acc = EventAccumulator(event_file)
                event_acc.Reload()
                tags = event_acc.Tags()['tensors']
                data_dict = {}
                for tag in tags:
                    data = event_acc.Tensors(tag)
                    data_array = np.concatenate([np.frombuffer(event.tensor_proto.tensor_content, dtype=np.float32) for event in data])
                    data_dict[tag] = data_array

                # Handle 'keras' separately
                if 'keras' in data_dict:
                    keras_array = data_dict.pop('keras')
                    data_array = np.column_stack(list(data_dict.values()))
                else:
                    data_array = np.column_stack(list(data_dict.values()))

                # Add a "step" column at the beginning of data_array
                num_steps = data_array.shape[0]
                step_array = np.arange(num_steps).reshape(num_steps, 1)
                data_array = np.hstack((step_array, data_array))

                # Get list of tag names including the new "step" column
                tag_names = ['step'] + list(data_dict.keys())
                
                # Add tag names as header for each column
                header = np.array(tag_names).reshape(1, -1)
                data_array = np.vstack((header, data_array))

                csv_file = os.path.join(csv_dir, f'{model_dir}_{subdir}.csv')
                np.savetxt(csv_file, data_array, delimiter=',', fmt='%s')
                print(f'Saved CSV file: {csv_file}')
