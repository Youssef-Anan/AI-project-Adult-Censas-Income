import pickle
import numpy as np

def load_label_encoders():
    """
    Load label encoders from saved file.
    Returns:
        dict: Loaded label encoders.
    """
    with open(r'AI project (adult census)\models\label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
    return label_encoders

def load_scalers():
    """
    Load scalers from saved file.
    Returns:
        dict: Loaded scalers.
    """
    with open(r'AI project (adult census)\models\scalers.pkl', 'rb') as file:
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


def group_education(level):
    """
    Group education levels as done in the training phase.
    """
    if level in ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']:
        return 'School'
    elif level in ['HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm']:
        return 'College'
    else:
        return 'After-graduate'
