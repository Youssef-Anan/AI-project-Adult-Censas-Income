from flask import Flask, render_template, request
import pickle
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
    'decision_tree': pickle.load(open(r'AI project (adult census)\models\decision_tree_model.pkl', 'rb')),
    'logistic_regression': pickle.load(open(r'AI project (adult census)\models\logistic_regression_model.pkl', 'rb')),
    'svm': pickle.load(open(r'AI project (adult census)\models\svm_model.pkl', 'rb')),
    'knn': pickle.load(open(r'AI project (adult census)\models\knn_model.pkl', 'rb'))
}

# Load X_test and Y_test
X_test = pickle.load(open(r'AI project (adult census)\models\X_test.pkl', 'rb'))
Y_test = pickle.load(open(r'AI project (adult census)\models\Y_test.pkl', 'rb'))

# Load encoders and scalers
label_encoders = load_label_encoders()
scalers = load_scalers()

# Define feature order as used during training
feature_order = ['age', 'education', 'marital.status',
                'occupation', 'relationship', 'race', 'sex', 'capital.gain',
                'capital.loss', 'hours.per.week']
FEATURE_MAPPING = {
    "Age": "age",
    "Education": "education",
    "Marital Status": "marital.status",
    "Occupation": "occupation",
    "Relationship": "relationship",
    "Race": "race",
    "Sex": "sex",
    "Capital Gain": "capital.gain",
    "Capital Loss": "capital.loss",
    "Hours Per Week": "hours.per.week",
}


@app.route('/')
def home():
    """
    Home page.
    """
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Prediction page.
    """
    feature_labels = list(FEATURE_MAPPING.keys())  # Display user-friendly names in form
    income_mapping = {0: "<=50K", 1: ">50K"}  # Add this mapping for income decoding

    if request.method == 'POST':
        try:
            # Collect user input for each feature
            inputs = request.form.to_dict()
            model_name = request.form.get('model')  # Assume there's a form field named 'model'
            # Standardize input keys using FEATURE_MAPPING
            standardized_inputs = standardize_input_keys(inputs, FEATURE_MAPPING)

            # Transform 'education' as done during training
            if 'education' in standardized_inputs:
                standardized_inputs['education'] = group_education(standardized_inputs['education'])

            # Transform 'marital.status' as done during training
            if 'marital.status' in standardized_inputs:
                marital_status_mapping = {
                    'Married-civ-spouse': 'Married',
                    'Married-AF-spouse': 'Married',
                    'Married-spouse-absent': 'Married',
                    'Never-married': 'Single',
                    'Divorced': 'Separated',
                    'Separated': 'Separated',
                    'Widowed': 'Separated'
                }
                standardized_inputs['marital.status'] = marital_status_mapping.get(
                    standardized_inputs['marital.status'],
                    standardized_inputs['marital.status']  # Default to the input if not in mapping
                )

            # Transform 'occupation' as done during training
            if 'occupation' in standardized_inputs:
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
                standardized_inputs['occupation'] = occupation_mapping.get(
                    standardized_inputs['occupation'],
                    standardized_inputs['occupation']  # Default to the input if not in mapping
                )

            # Validate the transformed input
            for feature, value in standardized_inputs.items():
                if feature in label_encoders and value not in label_encoders[feature].classes_:
                    raise ValueError(f"Invalid value '{value}' for feature '{feature}'.")

            # Preprocess input
            processed_input = preprocess_input(
                standardized_inputs,
                label_encoders,
                scalers,
                list(FEATURE_MAPPING.values())
            )
            preprocessed_input2 = np.array([[-0.55818081, 1.85605469, -0.05319823, 1.46098002, -0.21910053, 0.32692737, -2.21113531, 0.77086519, 0.82940558, 0.23561456]])


            # Predict
            model_choice = standardized_inputs.pop('model', model_name)
            if model_choice not in models:
                raise ValueError("Invalid model selection.")

            prediction = models[model_choice].predict(processed_input)
            probabilities = models[model_choice].predict_proba(processed_input)
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
    """
    Visualization page.
    """
    if request.method == 'POST':
        model_choice = request.form.get('model')
        if model_choice not in models:
            return render_template('visualize.html', error="Invalid model selection.")

        model = models[model_choice]
        decision_tree_plot, roc_curve_plot, confusion_matrix_plot = None, None, None

        # Decision Tree Visualization
        if model_choice == 'decision_tree':
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(
                model,
                filled=True,
                feature_names=feature_order,
                class_names=["<=50K", ">50K"],
                ax=ax
            )
            ax.set_title("Decision Tree")
            decision_tree_plot = generate_plot_base64(fig)

        # Confusion Matrix
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(Y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"]).plot(ax=ax)
        ax.set_title(f"{model_choice.capitalize()} - Confusion Matrix")
        confusion_matrix_plot = generate_plot_base64(fig)

        # ROC Curve
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(Y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
            ax.set_title(f"{model_choice.capitalize()} - ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
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
