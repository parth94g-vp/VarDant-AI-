# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import joblib

# # Load dataset
# df = pd.read_csv(r"D:\MIT\SEM 2\MP\Test_Project_1\fertilizer.csv")

# # Fix typo
# df.rename(columns={"Temparature": "Temperature"}, inplace=True)

# # Select improved feature set
# df = df[
#     [
#         "Soil Type",
#         "Crop Type",
#         "Nitrogen",
#         "Phosphorous",
#         "Potassium",
#         "Temperature",
#         "Humidity",
#         "Moisture",
#         "Fertilizer Name"
#     ]
# ]

# # Drop missing values
# df.dropna(inplace=True)

# # Encode categorical columns
# soil_encoder = LabelEncoder()
# crop_encoder = LabelEncoder()
# fertilizer_encoder = LabelEncoder()

# df["Soil Type"] = soil_encoder.fit_transform(df["Soil Type"])
# df["Crop Type"] = crop_encoder.fit_transform(df["Crop Type"])
# df["Fertilizer Name"] = fertilizer_encoder.fit_transform(df["Fertilizer Name"])

# # Split data
# X = df.drop("Fertilizer Name", axis=1)
# y = df["Fertilizer Name"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# # Stronger Random Forest
# model = RandomForestClassifier(
#     n_estimators=500,
#     max_depth=20,
#     class_weight="balanced",
#     random_state=42,
#     n_jobs=-1
# )

# # Train
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Improved Accuracy:", accuracy)

# # Save model & encoders
# joblib.dump(model, "fertilizer_model.pkl")
# joblib.dump(soil_encoder, "fertilizer_soil_encoder.pkl")
# joblib.dump(crop_encoder, "fertilizer_crop_encoder.pkl")
# joblib.dump(fertilizer_encoder, "fertilizer_encoder.pkl")

# print("✅ Improved Fertilizer Model Saved")

# print(df["Fertilizer Name"].value_counts())



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv(r"D:\MIT\SEM 2\MP\Test_Project_1\fertilizer.csv")

# Fix typo
df.rename(columns={"Temparature": "Temperature"}, inplace=True)

# Map fertilizer → category
def map_fertilizer_category(name):
    name = name.upper()

    if name in ["UREA"]:
        return "Nitrogen-rich"
    elif name in ["DAP", "SSP"]:
        return "Phosphorus-rich"
    elif name in ["MOP"]:
        return "Potassium-rich"
    elif "-" in name:
        return "Balanced"
    else:
        return "Organic"

df["Fertilizer Category"] = df["Fertilizer Name"].apply(map_fertilizer_category)

# Select features
df = df[
    [
        "Soil Type",
        "Crop Type",
        "Nitrogen",
        "Phosphorous",
        "Potassium",
        "Temperature",
        "Humidity",
        "Moisture",
        "Fertilizer Category"
    ]
]

df.dropna(inplace=True)

# Encode categorical columns
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
category_encoder = LabelEncoder()

df["Soil Type"] = soil_encoder.fit_transform(df["Soil Type"])
df["Crop Type"] = crop_encoder.fit_transform(df["Crop Type"])
df["Fertilizer Category"] = category_encoder.fit_transform(df["Fertilizer Category"])

# Split
X = df.drop("Fertilizer Category", axis=1)
y = df["Fertilizer Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Category Model Accuracy:", accuracy_score(y_test, y_pred))

# Save
joblib.dump(model, "fertilizer_category_model.pkl")
joblib.dump(soil_encoder, "fertilizer_soil_encoder.pkl")
joblib.dump(crop_encoder, "fertilizer_crop_encoder.pkl")
joblib.dump(category_encoder, "fertilizer_category_encoder.pkl")

print("✅ Fertilizer Category Model Saved")