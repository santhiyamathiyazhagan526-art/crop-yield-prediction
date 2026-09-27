import pandas as pd

file_path = "../dataset/Crop_Month_Season_Suitability.xlsx"

erode = pd.read_excel(
    file_path,
    sheet_name="Erode"
)

coimbatore = pd.read_excel(
    file_path,
    sheet_name="Coimbatore"
)

print("====================================")
print("CROP CALENDAR TEST")
print("====================================")

print("\nErode - January")

erode_january = erode[
    (erode["Month"] == "January") &
    (erode["Suitable_in_Cropping_Window"] == "Yes")
]

print(
    erode_january[
        [
            "District",
            "Crop",
            "Month",
            "Agro_Climatic_Period",
            "TNAU_Cropping_Window(s)"
        ]
    ].to_string(index=False)
)


print("\n====================================")

print("\nCoimbatore - January")

coimbatore_january = coimbatore[
    (coimbatore["Month"] == "January") &
    (coimbatore["Suitable_in_Cropping_Window"] == "Yes")
]

print(
    coimbatore_january[
        [
            "District",
            "Crop",
            "Month",
            "Agro_Climatic_Period",
            "TNAU_Cropping_Window(s)"
        ]
    ].to_string(index=False)
)