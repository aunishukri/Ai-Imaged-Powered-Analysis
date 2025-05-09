import h5py
import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def fix_and_load_model(model_path, save_fixed=True):
    """
    Loads a Keras model with incompatible 'groups' parameter in DepthwiseConv2D layers
    by removing the problematic parameter.
    
    Args:
        model_path (str): Path to the original h5 model file
        save_fixed (bool): Whether to save the fixed model file
        
    Returns:
        The loaded Keras model
    """
    print(f"Attempting to fix and load model from: {model_path}")
    
    # Open the h5 file
    with h5py.File(model_path, 'r') as f:
        # Get the model config
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        
        # Recursively remove 'groups' from all DepthwiseConv2D layers
        groups_found = []
        
        def remove_groups_param(config, path=""):
            if isinstance(config, dict):
                if config.get('class_name') == 'DepthwiseConv2D' and 'config' in config:
                    if 'groups' in config['config']:
                        current_path = f"{path}.{config.get('name', '')}" if path else config.get('name', '')
                        groups_found.append(current_path)
                        del config['config']['groups']
                        
                for key, value in config.items():
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(value, (dict, list)):
                        remove_groups_param(value, new_path)
            elif isinstance(config, list):
                for i, item in enumerate(config):
                    new_path = f"{path}[{i}]"
                    remove_groups_param(item, new_path)
        
        remove_groups_param(model_config)
        
        if groups_found:
            print(f"Removed 'groups' parameter from {len(groups_found)} DepthwiseConv2D layers:")
            for layer in groups_found:
                print(f"  - {layer}")
        else:
            print("No 'groups' parameters found to remove.")
        
        # Create a fixed model file
        if save_fixed:
            fixed_filename = os.path.splitext(model_path)[0] + "_fixed.h5"
            
            # Create model from the modified config
            model = tf.keras.models.model_from_json(json.dumps(model_config))
            
            # Load weights from original file
            # Create temporary weights dictionary
            weights_dict = {}
            for layer_name in f['model_weights']:
                if layer_name == 'keras_learning_phase':
                    continue
                layer_weights = []
                weight_names = []
                for weight_name in f['model_weights'][layer_name]:
                    if 'weight_names' in weight_name:
                        weight_names_data = f['model_weights'][layer_name][weight_name][...]
                        if isinstance(weight_names_data, bytes):
                            weight_names = json.loads(weight_names_data.decode('utf-8'))
                        else:
                            weight_names = [name.decode('utf-8') for name in weight_names_data]
                for weight_name in weight_names:
                    weight_path = f'model_weights/{layer_name}/{weight_name}'
                    if weight_path in f:
                        layer_weights.append(f[weight_path][...])
                if layer_weights:
                    weights_dict[layer_name] = layer_weights
            
            # Set weights to the model layer by layer
            for layer in model.layers: