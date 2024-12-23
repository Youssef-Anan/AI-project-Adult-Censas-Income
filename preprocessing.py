import pickle
import numpy as np
import os

def load_from_models_directory(filename):
    """
    Load a file from the models directory with a relative path.
    Args:
        filename (str): The name of the file to load.
    Returns:
        object: The loaded object from the file.
    """
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, 'models', filename)
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_label_encoders():
    """
    Load label encoders from saved file.
    Returns:
        dict: Loaded label encoders.
    """
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'label_encoders.pkl')
    with open(model_path, 'rb') as file:
        label_encoders = pickle.load(file)
    return label_encoders

def load_scalers():
    """
    Load scalers from saved file.
    Returns:
        dict: Loaded scalers.
    """
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'scalers.pkl')
    with open(model_path, 'rb') as file:
        scalers = pickle.load(file)
    return scalers

def preprocess_input(inputs, label_encoders, scalers, feature_order):
    """
    Preprocess user input for prediction.
    """
    # Apply mapping for 'marital.status'
    if 'marital.status' in inputs:
        marital_status_mapping = {
            'Married-civ-spouse': 'Married',
            'Married-AF-spouse': 'Married',
            'Married-spouse-absent': 'Married',
            'Never-married': 'Single',
            'Divorced': 'Separated',
            'Separated': 'Separated',
            'Widowed': 'Separated'
        }
        inputs['marital.status'] = marital_status_mapping.get(inputs['marital.status'], inputs['marital.status'])

    # Process other features as usual
    processed_inputs = []
    for feature in feature_order:
        if feature in label_encoders:
            # Encode categorical features
            value = inputs.get(feature)
            if value not in label_encoders[feature].classes_:
                raise ValueError(f"Invalid value '{value}' for feature '{feature}'.")
            encoded_value = label_encoders[feature].transform([value])[0]
            processed_inputs.append(encoded_value)
        elif feature in scalers:
            # Scale numerical features
            value = float(inputs.get(feature, 0))
            scaled_value = scalers[feature].transform([[value]])[0, 0]
            processed_inputs.append(scaled_value)
        else:
            # Handle raw numerical values
            value = float(inputs.get(feature, 0))
            processed_inputs.append(value)

    return np.array(processed_inputs).reshape(1, -1)

def decode_output(encoded_output, label_encoders, target_column):
    """
    Decode the output of the model prediction.
    Args:
        encoded_output (int): Encoded output value.
        label_encoders (dict): Dictionary of label encoders.
        target_column (str): Name of the target column.
    Returns:
        str: Decoded label.
    """
    if target_column in label_encoders:
        return label_encoders[target_column].inverse_transform([encoded_output])[0]
    else:
        raise ValueError(f"Target column '{target_column}' is not in label encoders.")

def standardize_input_keys(inputs, feature_mapping):
    """
    Map input dictionary keys to match the feature_order using FEATURE_MAPPING.
    - inputs: Dictionary of user inputs.
    - feature_mapping: Dictionary mapping user-friendly names to technical names.
    """
    standardized_inputs = {}
    for user_key, technical_key in feature_mapping.items():
        if user_key in inputs:
            standardized_inputs[technical_key] = inputs[user_key]
        else:
            raise ValueError(f"Feature '{technical_key}' is missing from input data.")
    return standardized_inputs

def map_education(level):
    """
    Group education levels as done in the training phase.
    """
    if level in ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']:
        return 'School'
    elif level in ['HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm']:
        return 'College'
    else:
        return 'After-graduate'

def map_marital_status(status):
    """
    Map marital statuses into broader categories.
    """
    marital_status_mapping = {
        'Married-civ-spouse': 'Married',
        'Married-AF-spouse': 'Married',
        'Married-spouse-absent': 'Married',
        'Never-married': 'Single',
        'Divorced': 'Separated',
        'Separated': 'Separated',
        'Widowed': 'Separated'
    }
    return marital_status_mapping.get(status, status)

def map_occupation(occupation):
    """
    Map occupations into broader categories.
    """
    occupation_mapping = {
        'Prof-specialty': 'Professional',
        'Exec-managerial': 'Professional',
        'Tech-support': 'Professional',
        'Adm-clerical': 'Sales',
        'Sales': 'Sales',
        'Other-service': 'Others',
        'Protective-serv': 'Others',
        'Priv-house-serv': 'Others',
        'Armed-Forces': 'Others',
        'Craft-repair': 'Blue-Collar',
        'Machine-op-inspct': 'Blue-Collar',
        'Transport-moving': 'Blue-Collar',
        'Handlers-cleaners': 'Blue-Collar',
        'Farming-fishing': 'Blue-Collar'
    }
    return occupation_mapping.get(occupation, occupation)