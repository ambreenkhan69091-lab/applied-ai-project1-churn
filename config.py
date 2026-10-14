# Constants shared across the app

RAW_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["Fiber optic", "DSL", "No"]
TIERS = ["Stable", "Watchlist", "High risk"]

COLORS = {
    "high": "#ef4444",
    "mid": "#f59e0b",
    "low": "#22c55e",
    "accent": "#38bdf8",
    "violet": "#a78bfa",
    "muted": "#8892b0",
}

# the model was trained on a USD price list, so USD is the internal currency
# and PKR is only used for display / input
DEFAULT_FX = 280.0
BILL_MIN_USD = 18.0
BILL_MAX_USD = 120.0

# a "typical" customer, used as the baseline when explaining a prediction
REFERENCE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 32,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "One year", "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 65.0, "TotalCharges": 32 * 65.0,
}

ATTRIBUTION_GROUPS = {
    "Tenure": ["tenure"],
    "Monthly bill": ["MonthlyCharges"],
    "Contract type": ["Contract"],
    "Payment method": ["PaymentMethod"],
    "Paperless billing": ["PaperlessBilling"],
    "Internet & add-ons": ["InternetService"] + SERVICES,
    "Phone service": ["PhoneService", "MultipleLines"],
    "Senior citizen": ["SeniorCitizen"],
    "Partner & dependents": ["Partner", "Dependents"],
    "Gender": ["gender"],
}

PRESETS = {
    "Custom": None,
    "High-risk archetype": dict(
        tenure=3, contract="Month-to-month", monthly=95.0, internet="Fiber optic", services=[],
        phone="Yes", multi="Yes", payment="Electronic check", paperless="Yes",
    ),
    "Loyal archetype": dict(
        tenure=60, contract="Two year", monthly=55.0, internet="DSL",
        services=["OnlineSecurity", "TechSupport", "OnlineBackup"], phone="Yes", multi="No",
        payment="Credit card (automatic)", paperless="No",
    ),
    "Price-sensitive student": dict(
        tenure=8, contract="Month-to-month", monthly=35.0, internet="DSL", services=[],
        phone="Yes", multi="No", payment="Mailed check", paperless="No",
    ),
}

# city -> (lat, lon, share of customers in the demo data)
CITIES = {
    "Karachi": (24.86, 67.01, 0.22),
    "Lahore": (31.55, 74.34, 0.20),
    "Islamabad": (33.68, 73.05, 0.08),
    "Rawalpindi": (33.60, 73.04, 0.08),
    "Faisalabad": (31.42, 73.08, 0.08),
    "Peshawar": (34.01, 71.58, 0.08),
    "Multan": (30.20, 71.47, 0.06),
    "Hyderabad": (25.39, 68.37, 0.05),
    "Quetta": (30.18, 66.99, 0.05),
    "Abbottabad": (34.15, 73.21, 0.04),
    "Sialkot": (32.49, 74.53, 0.03),
    "Gujranwala": (32.16, 74.19, 0.03),
}
