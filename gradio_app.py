import gradio as gr
import joblib
import numpy as np

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(
    "loan_approval_pipeline.pkl"
)

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_loan(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
):

    try:

        # Input Array
        data = np.array([[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]])

        # Prediction
        prediction = model.predict(data)

        # Probability
        probability = model.predict_proba(data)

        confidence = round(
            np.max(probability) * 100,
            2
        )

        # Result
        if prediction[0] == 1:

            return f"""
            ✅ Loan Approved

            Confidence: {confidence}%
            """

        return f"""
        ❌ Loan Rejected

        Confidence: {confidence}%
        """

    except Exception as e:

        return str(e)

# =====================================================
# CREATE INTERFACE
# =====================================================

interface = gr.Interface(

    fn=predict_loan,

    inputs=[

        gr.Number(label="Number of Dependents"),

        gr.Dropdown(
            choices=[0,1],
            label="Education (0=Graduate, 1=Not Graduate)"
        ),

        gr.Dropdown(
            choices=[0,1],
            label="Self Employed (0=No, 1=Yes)"
        ),

        gr.Number(label="Annual Income"),

        gr.Number(label="Loan Amount"),

        gr.Number(label="Loan Term"),

        gr.Number(label="CIBIL Score"),

        gr.Number(label="Residential Assets Value"),

        gr.Number(label="Commercial Assets Value"),

        gr.Number(label="Luxury Assets Value"),

        gr.Number(label="Bank Asset Value")

    ],

    outputs="text",

    title="🏦 Loan Approval Prediction System",

    description="""
    Enter Applicant Details For Prediction
    """,

    theme="soft"
)

# =====================================================
# LAUNCH APP
# =====================================================

interface.launch()