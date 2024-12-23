from flask import Flask, render_template, request
from preprocessing import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import io
import base64
import numpy as np
from sklearn.tree import plot_tree

# Initialize Flask app
app = Flask(__name__)

# Helper function to encode plots as Base64
def generate_plot_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return plot_data

# Load trained models
models = {
    'decision_tree': load_from_models_directory('decision_tree_model.pkl'),
    'logistic_regression': load_from_models_directory('logistic_regression_model.pkl'),
    'svm': load_from_models_directory('svm_model.pkl'),
    'knn': load_from_models_directory('knn_model.pkl')
}

# Load X_test and Y_test
X_test = load_from_models_directory('X_test.pkl')
Y_test = load_from_models_directory('Y_test.pkl')

# Load encoders and scalers
label_encoders = load_label_encoders()
scalers = load_scalers()

# Define feature order as used during training
feature_order = ['age', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week']

FEATURE_MAPPING = {"Age": "age", "Education": "education", "Marital Status": "marital.status", "Occupation": "occupation", "Relationship": "relationship", "Race": "race", "Sex": "sex", "Capital Gain": "capital.gain", "Capital Loss": "capital.loss", "Hours Per Week": "hours.per.week",}


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    feature_labels = list(FEATURE_MAPPING.keys())  # Display user-friendly names in form
    income_mapping = {0: "<=50K", 1: ">50K"}  # Add this mapping for income decoding

    if request.method == 'POST':
        try:
            # Collect user input for each feature
            inputs = request.form.to_dict()
            # Collect user input for model
            model_name = request.form.get('model')
            # Standardize input keys using FEATURE_MAPPING
            standardized_inputs = standardize_input_keys(inputs, FEATURE_MAPPING)

            # Apply mappings for categorical features
            if 'education' in standardized_inputs:
                standardized_inputs['education'] = map_education(standardized_inputs['education'])
            if 'marital.status' in standardized_inputs:
                standardized_inputs['marital.status'] = map_marital_status(standardized_inputs['marital.status'])
            if 'occupation' in standardized_inputs:
                standardized_inputs['occupation'] = map_occupation(standardized_inputs['occupation'])

            # Validate the transformed input
            for feature, value in standardized_inputs.items():
                if feature in label_encoders and value not in label_encoders[feature].classes_:
                    raise ValueError(f"Invalid value '{value}' for feature '{feature}'.")

            # Preprocess input
            processed_input = preprocess_input(
                standardized_inputs,            #Raw inputs entered by the user.
                label_encoders,                 #Used to transform string-based categorical features (e.g., education: "College") into numerical format (e.g., education: 1).
                scalers,                        #Scales numerical inputs (e.g., age, capital.gain) to a standardized range (e.g., between 0 and 1).
                list(FEATURE_MAPPING.values())  #A list of technical feature names in the correct order expected by the model.
            )

            # Prediction
            model_choice = standardized_inputs.pop('model', model_name) #Extractring the model data.
            if model_choice not in models:
                raise ValueError("Invalid model selection.")
            # Predict
            prediction = models[model_choice].predict(processed_input)
            # Decode prediction
            predicted_label = income_mapping.get(prediction[0], "Unknown")

            return render_template('result.html', prediction=predicted_label)

        except Exception as e:
            return render_template(
                'predict.html',
                feature_labels=feature_labels
            )

    return render_template('predict.html', feature_labels=feature_labels)



@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    if request.method == 'POST':
        model_choice = request.form.get('model')
        if model_choice not in models:
            return render_template('visualize.html', error="Invalid model selection.")

        model = models[model_choice]
        decision_tree_plot, roc_curve_plot, confusion_matrix_plot = None, None, None

        # Decision Tree Visualization
        if model_choice == 'decision_tree':
            # Create a new figure and axes for plotting the decision tree
            fig, ax = plt.subplots(figsize=(20, 10))
            
            # Plot the decision tree using the model, feature names, and class names
            plot_tree(
                model,  # The decision tree model to visualize
                filled=True,  # Fill nodes with colors representing the class
                feature_names=feature_order,  # List of feature names for labeling the tree
                class_names=["<=50K", ">50K"],  # Class labels for the target variable
                ax=ax  # Axes object where the tree will be plotted
            )
            
            # Set the title for the decision tree plot
            ax.set_title("Decision Tree")
            
            # Encode the decision tree plot as a Base64 string for embedding in the HTML response
            decision_tree_plot = generate_plot_base64(fig)

        # Confusion Matrix
        # Generate predictions on the test data using the selected model
        y_pred = model.predict(X_test)

        # Create a new figure and axes for the confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))

        # Compute the confusion matrix based on true and predicted labels
        cm = confusion_matrix(Y_test, y_pred)

        # Display the confusion matrix with labels for the target classes
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"]).plot(ax=ax)

        # Set the title for the confusion matrix plot
        ax.set_title(f"{model_choice.capitalize()} - Confusion Matrix")

        # Encode the confusion matrix plot as a Base64 string for embedding in the HTML response
        confusion_matrix_plot = generate_plot_base64(fig)

        # ROC Curve
        # Check if the model supports probability predictions
        if hasattr(model, "predict_proba"):
            # Compute the probabilities for the positive class (">50K") using the test data
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Compute the false positive rate (FPR), true positive rate (TPR), and thresholds for the ROC curve
            fpr, tpr, _ = roc_curve(Y_test, y_proba)
            
            # Calculate the Area Under the Curve (AUC) for the ROC curve
            roc_auc = auc(fpr, tpr)
            
            # Create a new figure and axes for the ROC curve plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot the ROC curve
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")  # Plot the FPR and TPR values with AUC as a label
            
            # Add a diagonal reference line representing a random classifier
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
            
            # Set the title and labels for the ROC curve plot
            ax.set_title(f"{model_choice.capitalize()} - ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            
            # Add a legend to the plot
            ax.legend(loc="lower right")
            
            # Encode the ROC curve plot as a Base64 string for embedding in the HTML response
            roc_curve_plot = generate_plot_base64(fig)


        return render_template(
            'visualize.html',
            decision_tree_plot=decision_tree_plot,
            confusion_matrix_plot=confusion_matrix_plot,
            roc_curve_plot=roc_curve_plot
        )
    return render_template('visualize.html')

if __name__ == '__main__':
    app.run(debug=True)
