import pandas as pd
import joblib
import os


# ============================================================
# CROP MONTH-WISE YIELD RECOMMENDATION SYSTEM
# ============================================================
#
# USER INPUT:
#   1. District
#   2. Month
#   3. Crop
#
# SYSTEM AUTOMATICALLY GETS:
#   - Rainfall
#   - Temperature
#   - Soil pH
#   - Fertilizer
#
# from the original historical crop-yield dataset.
#
# ============================================================


# ============================================================
# 1. PROJECT PATH
# ============================================================

# Current file:
# C:\CropYieldPrediction\model\month_crop_recommendation.py
#
# Project folder:
# C:\CropYieldPrediction

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# 2. FILE PATHS
# ============================================================

# Month-wise suitability Excel file
SUITABILITY_FILE = os.path.join(
    PROJECT_DIR,
    "dataset",
    "Crop_Month_Season_Suitability.xlsx"
)


# Original historical dataset
ORIGINAL_DATASET = os.path.join(
    PROJECT_DIR,
    "dataset",
    "Crop_Yield_Dataset_Coimbatore_Erode_500_Records.xlsx"
)


# Trained ML model
MODEL_FILE = os.path.join(
    PROJECT_DIR,
    "saved_model",
    "linear_regression_model.pkl"
)


# Preprocessor
PREPROCESSOR_FILE = os.path.join(
    PROJECT_DIR,
    "saved_model",
    "preprocessor.pkl"
)


# ============================================================
# 3. CHECK FILES
# ============================================================

print("\n==============================================")
print("      CROP MONTH-WISE YIELD RECOMMENDATION")
print("==============================================")


print("\nChecking project files...")


required_files = {
    "Suitability Excel": SUITABILITY_FILE,
    "Original Dataset": ORIGINAL_DATASET,
    "ML Model": MODEL_FILE,
    "Preprocessor": PREPROCESSOR_FILE
}


for file_name, file_path in required_files.items():

    if not os.path.exists(file_path):

        print(f"\nERROR: {file_name} not found!")

        print("Expected location:")
        print(file_path)

        exit()

    else:

        print(f"✓ {file_name} found")


# ============================================================
# 4. LOAD SUITABILITY DATA
# ============================================================

try:

    suitability_sheets = pd.read_excel(
        SUITABILITY_FILE,
        sheet_name=None
    )

except Exception as e:

    print("\nERROR while loading suitability Excel:")
    print(e)

    exit()


# ============================================================
# 5. LOAD ORIGINAL DATASET
# ============================================================

try:

    original_data = pd.read_excel(
        ORIGINAL_DATASET,
        sheet_name=None
    )

except Exception as e:

    print("\nERROR while loading original dataset:")
    print(e)

    exit()


# ============================================================
# 6. COMBINE ORIGINAL DATASET SHEETS
# ============================================================

dataset_list = []


for sheet_name, sheet_data in original_data.items():

    if isinstance(sheet_data, pd.DataFrame):

        dataset_list.append(
            sheet_data
        )


if len(dataset_list) == 0:

    print("\nERROR: No data found in original dataset.")

    exit()


historical_data = pd.concat(
    dataset_list,
    ignore_index=True
)


# ============================================================
# 7. CLEAN ORIGINAL DATA
# ============================================================

historical_data["District"] = (
    historical_data["District"]
    .astype(str)
    .str.strip()
)

historical_data["Crop"] = (
    historical_data["Crop"]
    .astype(str)
    .str.strip()
)

historical_data["Fertilizer"] = (
    historical_data["Fertilizer"]
    .astype(str)
    .str.strip()
)


# Convert numerical columns
numeric_columns = [
    "Rainfall_mm",
    "Temperature_C",
    "Soil_pH"
]


for column in numeric_columns:

    historical_data[column] = pd.to_numeric(
        historical_data[column],
        errors="coerce"
    )


# ============================================================
# 8. LOAD MODEL
# ============================================================

try:

    model = joblib.load(
        MODEL_FILE
    )

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

except Exception as e:

    print("\nERROR while loading ML model:")
    print(e)

    exit()


print("✓ Historical dataset loaded")
print("✓ Machine Learning model loaded")


# ============================================================
# 9. MONTH NAME CONVERSION
# ============================================================

month_names = {

    "1": "January",
    "2": "February",
    "3": "March",
    "4": "April",
    "5": "May",
    "6": "June",
    "7": "July",
    "8": "August",
    "9": "September",
    "10": "October",
    "11": "November",
    "12": "December",

    "jan": "January",
    "january": "January",

    "feb": "February",
    "february": "February",

    "mar": "March",
    "march": "March",

    "apr": "April",
    "april": "April",

    "may": "May",

    "jun": "June",
    "june": "June",

    "jul": "July",
    "july": "July",

    "aug": "August",
    "august": "August",

    "sep": "September",
    "sept": "September",
    "september": "September",

    "oct": "October",
    "october": "October",

    "nov": "November",
    "november": "November",

    "dec": "December",
    "december": "December"
}


# ============================================================
# 10. DISTRICT ALIASES
# ============================================================

district_names = {

    "erode": "Erode",
    "erod": "Erode",

    "coimbatore": "Coimbatore",
    "coimatore": "Coimbatore",
    "coimbator": "Coimbatore",
    "coimbatoree": "Coimbatore",
    "kovai": "Coimbatore"
}


# ============================================================
# 11. CROP ALIASES
# ============================================================

crop_names = {

    "rice": "Rice",
    "paddy": "Rice",

    "sugarcane": "Sugarcane",
    "sugar cane": "Sugarcane",

    "turmeric": "Turmeric",

    "banana": "Banana",

    "groundnut": "Groundnut",
    "ground nut": "Groundnut",
    "peanut": "Groundnut",

    "cotton": "Cotton",

    "maize": "Maize",
    "corn": "Maize",

    "tapioca": "Tapioca",
    "cassava": "Tapioca",

    "ragi": "Ragi"
}


# ============================================================
# 12. FUNCTION:
# GET HISTORICAL ENVIRONMENTAL CONDITIONS
# ============================================================

def get_environmental_values(
    district,
    crop
):

    # Filter historical records
    crop_data = historical_data[
        (
            historical_data["District"].str.lower()
            == district.lower()
        )
        &
        (
            historical_data["Crop"].str.lower()
            == crop.lower()
        )
    ].copy()


    # --------------------------------------------------------
    # If crop-specific data is available
    # --------------------------------------------------------

    if not crop_data.empty:

        rainfall = crop_data[
            "Rainfall_mm"
        ].mean()

        temperature = crop_data[
            "Temperature_C"
        ].mean()

        soil_ph = crop_data[
            "Soil_pH"
        ].mean()


        # Most frequently used fertilizer
        fertilizer_mode = (
            crop_data["Fertilizer"]
            .dropna()
            .mode()
        )


        if not fertilizer_mode.empty:

            fertilizer = fertilizer_mode.iloc[0]

        else:

            fertilizer = "Urea"


        return (
            rainfall,
            temperature,
            soil_ph,
            fertilizer
        )


    # --------------------------------------------------------
    # If crop-specific data unavailable,
    # use district-level values
    # --------------------------------------------------------

    district_data = historical_data[
        historical_data["District"].str.lower()
        == district.lower()
    ].copy()


    if district_data.empty:

        return (
            historical_data["Rainfall_mm"].mean(),
            historical_data["Temperature_C"].mean(),
            historical_data["Soil_pH"].mean(),
            "Urea"
        )


    rainfall = district_data[
        "Rainfall_mm"
    ].mean()

    temperature = district_data[
        "Temperature_C"
    ].mean()

    soil_ph = district_data[
        "Soil_pH"
    ].mean()


    fertilizer_mode = (
        district_data["Fertilizer"]
        .dropna()
        .mode()
    )


    if not fertilizer_mode.empty:

        fertilizer = fertilizer_mode.iloc[0]

    else:

        fertilizer = "Urea"


    return (
        rainfall,
        temperature,
        soil_ph,
        fertilizer
    )


# ============================================================
# 13. FUNCTION:
# PREDICT CROP YIELD
# ============================================================

def predict_yield(
    crop,
    district
):

    # --------------------------------------------------------
    # Automatically obtain environmental values
    # --------------------------------------------------------

    (
        rainfall,
        temperature,
        soil_ph,
        fertilizer
    ) = get_environmental_values(
        district,
        crop
    )


    # --------------------------------------------------------
    # Create model input
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "Crop": [crop],

        "District": [district],

        "Rainfall_mm": [rainfall],

        "Temperature_C": [temperature],

        "Soil_pH": [soil_ph],

        "Fertilizer": [fertilizer]

    })


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    transformed_data = preprocessor.transform(
        input_data
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_yield = model.predict(
        transformed_data
    )[0]


    # Prevent negative prediction
    predicted_yield = max(
        0,
        predicted_yield
    )


    return predicted_yield


# ============================================================
# 14. DISTRICT INPUT
# ============================================================

print("\n----------------------------------------------")
print("AVAILABLE DISTRICTS")
print("----------------------------------------------")

print("1. Erode")
print("2. Coimbatore")


district_input = input(
    "\nEnter District: "
).strip().lower()


if district_input not in district_names:

    print("\nERROR: Invalid district.")

    print(
        "Please enter Erode or Coimbatore."
    )

    exit()


district = district_names[
    district_input
]


# ============================================================
# 15. MONTH INPUT
# ============================================================

print("\n----------------------------------------------")
print("SELECT MONTH")
print("----------------------------------------------")


month_input = input(
    "Enter Month (January / Jan / 1): "
).strip().lower()


if month_input not in month_names:

    print("\nERROR: Invalid month.")

    exit()


month = month_names[
    month_input
]


# ============================================================
# 16. LOAD DISTRICT SUITABILITY SHEET
# ============================================================

if district not in suitability_sheets:

    print(
        f"\nERROR: No suitability data available "
        f"for {district}."
    )

    exit()


district_data = suitability_sheets[
    district
].copy()


# ============================================================
# 17. CLEAN SUITABILITY DATA
# ============================================================

district_data["Crop"] = (
    district_data["Crop"]
    .astype(str)
    .str.strip()
)

district_data["Month"] = (
    district_data["Month"]
    .astype(str)
    .str.strip()
)

district_data[
    "Suitable_in_Cropping_Window"
] = (
    district_data[
        "Suitable_in_Cropping_Window"
    ]
    .astype(str)
    .str.strip()
)


# ============================================================
# 18. DISPLAY AVAILABLE CROPS
# ============================================================

available_crops = sorted(
    district_data["Crop"]
    .dropna()
    .unique()
)


print("\n----------------------------------------------")
print(
    f"AVAILABLE CROPS FOR {district.upper()}"
)
print("----------------------------------------------")


for crop in available_crops:

    print("✓", crop)


# ============================================================
# 19. CROP INPUT
# ============================================================

crop_input = input(
    "\nEnter Crop to check: "
).strip().lower()


if crop_input not in crop_names:

    print("\nERROR: Invalid crop.")

    print("\nAvailable crops:")

    for crop in available_crops:

        print(" -", crop)

    exit()


selected_crop = crop_names[
    crop_input
]


# ============================================================
# 20. CHECK SELECTED CROP + MONTH
# ============================================================

selected_crop_data = district_data[
    (
        district_data["Crop"].str.lower()
        == selected_crop.lower()
    )
    &
    (
        district_data["Month"].str.lower()
        == month.lower()
    )
]


# ============================================================
# 21. DETERMINE SUITABILITY
# ============================================================

if selected_crop_data.empty:

    suitability = "no"

else:

    suitability = str(
        selected_crop_data.iloc[0][
            "Suitable_in_Cropping_Window"
        ]
    ).strip().lower()


# ============================================================
# 22. IF CROP IS SUITABLE
# ============================================================

if suitability == "yes":

    print("\n==============================================")
    print("              CROP SUITABILITY")
    print("==============================================")

    print(
        f"District : {district}"
    )

    print(
        f"Month    : {month}"
    )

    print(
        f"Crop     : {selected_crop}"
    )


    print("\nResult : YES")


    print(
        f"\n✓ {selected_crop} is suitable for "
        f"{month} in {district}."
    )


    # --------------------------------------------------------
    # AUTOMATIC PREDICTION
    # --------------------------------------------------------

    predicted_yield = predict_yield(
        selected_crop,
        district
    )


    # --------------------------------------------------------
    # DISPLAY YIELD
    # --------------------------------------------------------

    print("\n==============================================")
    print("             EXPECTED CROP YIELD")
    print("==============================================")


    print(
        f"Crop : {selected_crop}"
    )


    print(
        f"Expected Yield : "
        f"{predicted_yield:.2f} tons/hectare"
    )


    print("\n==============================================")
    print("              RECOMMENDATION")
    print("==============================================")


    print(
        f"✓ {selected_crop} can be considered "
        f"for cultivation in {month}."
    )


    print(
        f"✓ Expected yield: "
        f"{predicted_yield:.2f} tons/hectare"
    )


    print("==============================================")


# ============================================================
# 23. IF CROP IS NOT SUITABLE
# ============================================================

else:

    print("\n==============================================")
    print("              CROP SUITABILITY")
    print("==============================================")


    print(
        f"District : {district}"
    )

    print(
        f"Month    : {month}"
    )

    print(
        f"Crop     : {selected_crop}"
    )


    print("\nResult : NO")


    print(
        f"\n✗ {selected_crop} is not recommended "
        f"for {month} in {district}."
    )


    # ========================================================
    # FIND SUITABLE ALTERNATIVE CROPS
    # ========================================================

    suitable_data = district_data[
        (
            district_data["Month"].str.lower()
            == month.lower()
        )
        &
        (
            district_data[
                "Suitable_in_Cropping_Window"
            ]
            .str.lower()
            == "yes"
        )
    ].copy()


    # Remove selected crop
    suitable_data = suitable_data[
        suitable_data["Crop"].str.lower()
        != selected_crop.lower()
    ]


    # Remove duplicates
    alternative_crops = (
        suitable_data["Crop"]
        .dropna()
        .unique()
        .tolist()
    )


    # ========================================================
    # NO ALTERNATIVE
    # ========================================================

    if len(alternative_crops) == 0:

        print(
            f"\nNo suitable alternative crops "
            f"were found for {month} in {district}."
        )

        exit()


    # ========================================================
    # PREDICT ALTERNATIVE CROP YIELDS
    # ========================================================

    prediction_results = []


    for crop in alternative_crops:

        predicted_yield = predict_yield(
            crop,
            district
        )


        prediction_results.append({

            "Crop": crop,

            "Predicted Yield": predicted_yield
        })


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        prediction_results
    )


    # Sort highest yield first
    results_df = results_df.sort_values(
        by="Predicted Yield",
        ascending=False
    ).reset_index(
        drop=True
    )


    # ========================================================
    # DISPLAY ALTERNATIVES
    # ========================================================

    print("\n==============================================")
    print("          SUITABLE ALTERNATIVE CROPS")
    print("==============================================")


    for index, row in results_df.iterrows():

        print(
            f"{index + 1}. "
            f"{row['Crop']} → "
            f"{row['Predicted Yield']:.2f} "
            f"tons/hectare"
        )


    # ========================================================
    # BEST ALTERNATIVE
    # ========================================================

    best_crop = results_df.iloc[0][
        "Crop"
    ]


    best_yield = results_df.iloc[0][
        "Predicted Yield"
    ]


    print("\n==============================================")
    print("             BEST ALTERNATIVE")
    print("==============================================")


    print(
        f"Recommended Crop : {best_crop}"
    )


    print(
        f"Expected Yield   : "
        f"{best_yield:.2f} tons/hectare"
    )


    print("\n----------------------------------------------")


    print(
        f"✓ {selected_crop} is not suitable "
        f"for {month} in {district}."
    )


    print(
        f"✓ {best_crop} is the highest-yielding "
        f"alternative among the suitable crops."
    )


    print(
        f"✓ Predicted yield: "
        f"{best_yield:.2f} tons/hectare"
    )


    print("==============================================")