# %% Here we define the model metatadata we want to integrate in the STAC collection

from pathlib import Path

import pandas as pd

# Define comprehensive bounding boxes for each region
REGION_BBOXES = {
    "Africa": {
        "bbox": [-25.0, -35.0, 55.0, 40.0],  # [min_lon, min_lat, max_lon, max_lat]
        "description": "Covers continental Africa including Madagascar",
        "countries_included": [
            "Algeria",
            "Angola",
            "Benin",
            "Botswana",
            "Burkina Faso",
            "Burundi",
            "Cabo Verde",
            "Cameroon",
            "Central African Republic",
            "Chad",
            "Comoros",
            "Congo",
            "DR Congo",
            "Djibouti",
            "Egypt",
            "Equatorial Guinea",
            "Eritrea",
            "Eswatini",
            "Ethiopia",
            "Gabon",
            "Gambia",
            "Ghana",
            "Guinea",
            "Guinea-Bissau",
            "Ivory Coast",
            "Kenya",
            "Lesotho",
            "Liberia",
            "Libya",
            "Madagascar",
            "Malawi",
            "Mali",
            "Mauritania",
            "Mauritius",
            "Morocco",
            "Mozambique",
            "Namibia",
            "Niger",
            "Nigeria",
            "Rwanda",
            "São Tomé and Príncipe",
            "Senegal",
            "Seychelles",
            "Sierra Leone",
            "Somalia",
            "South Africa",
            "South Sudan",
            "Sudan",
            "Tanzania",
            "Togo",
            "Tunisia",
            "Uganda",
            "Zambia",
            "Zimbabwe",
        ],
    },
    "Southeast Asia": {
        "bbox": [90.0, -10.0, 150.0, 30.0],
        "description": "Covers mainland and maritime Southeast Asia",
        "countries_included": [
            "Myanmar",
            "Thailand",
            "Laos",
            "Vietnam",
            "Cambodia",
            "Malaysia",
            "Singapore",
            "Indonesia",
            "Philippines",
            "Brunei",
            "East Timor",
            "Papua New Guinea",
            "Parts of southern China",
        ],
    },
    "Latin America": {
        "bbox": [-85.0, -60.0, -30.0, 15.0],
        "description": "Covers Central and South America, Caribbean",
        "countries_included": [
            "Guatemala",
            "Belize",
            "El Salvador",
            "Honduras",
            "Nicaragua",
            "Costa Rica",
            "Panama",
            "Colombia",
            "Venezuela",
            "Guyana",
            "Suriname",
            "French Guiana",
            "Brazil",
            "Ecuador",
            "Peru",
            "Bolivia",
            "Paraguay",
            "Chile",
            "Argentina",
            "Uruguay",
            "Cuba",
            "Jamaica",
            "Haiti",
            "Dominican Republic",
            "Puerto Rico",
            "Bahamas",
            "Trinidad and Tobago",
        ],
    },
}


def create_enhanced_model_metadata():
    """Create model metadata with proper bounding boxes and updated model URLs"""

    models_data = []

    # Africa Model (your updated version)
    models_data.append(
        {
            # Basic Identification
            "model_id": "WorldAgriCommodities_Africa_v1",
            "model_name": "Africa Crop Classification 2024",
            "region": "Africa",
            "description": "Deep learning model for crop classification across Africa",
            # Spatial Information
            "bbox": str(REGION_BBOXES["Africa"]["bbox"]),
            "bbox_min_lon": REGION_BBOXES["Africa"]["bbox"][0],
            "bbox_min_lat": REGION_BBOXES["Africa"]["bbox"][1],
            "bbox_max_lon": REGION_BBOXES["Africa"]["bbox"][2],
            "bbox_max_lat": REGION_BBOXES["Africa"]["bbox"][3],
            "crs": "EPSG:4326",
            "countries_covered": REGION_BBOXES["Africa"]["countries_included"],
            # Technical Specifications
            "framework": "ONNX",
            "input_shape": 17,
            "input_channels": [
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B11",
                "B12",
                "NDVI",
                "NDRE",
                "EVI",
                "VV",
                "VH",
                "DEM",
                "lon",
                "lat",
            ],
            "output_shape": 25,
            "output_classes": [
                "Background",
                "Other_large_scale_cropland",
                "Pasture",
                "Mining",
                "Other_small_scale_cropland",
                "Roads",
                "Forest",
                "Plantation_forest",
                "Coffee",
                "Build_up",
                "Water",
                "Oil_palm",
                "Rubber",
                "Cacao",
                "Avocado",
                "Soy",
                "Sugar",
                "Maize",
                "Banana",
                "Pineapple",
                "Rice",
                "Wood_logging",
                "Cashew",
                "Tea",
                "Others",
            ],
            # Temporal Information
            "temporal_start": "2020-01-01",
            "temporal_end": "2030-12-31",
            # Storage Information
            "model_url": "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/WorldAgriCommodities/models/best_weights_att_unet_lagtime_5_Fused3_2023_totalLoss6V1_without_loss_sentAfrica6.onnx",
            # Performance Metrics
            "overall_accuracy": "None",
            "f1_score_macro": "None",
        }
    )

    # Southeast Asia Model (updated to match your structure)
    models_data.append(
        {
            # Basic Identification
            "model_id": "WorldAgriCommodities_SoutheastAsia_v1",
            "model_name": "Southeast Asia Crop Classification 2024",
            "region": "Southeast Asia",
            "description": "Deep learning model for crop classification across Southeast Asia",
            # Spatial Information
            "bbox": str(REGION_BBOXES["Southeast Asia"]["bbox"]),
            "bbox_min_lon": REGION_BBOXES["Southeast Asia"]["bbox"][0],
            "bbox_min_lat": REGION_BBOXES["Southeast Asia"]["bbox"][1],
            "bbox_max_lon": REGION_BBOXES["Southeast Asia"]["bbox"][2],
            "bbox_max_lat": REGION_BBOXES["Southeast Asia"]["bbox"][3],
            "crs": "EPSG:4326",
            "countries_covered": REGION_BBOXES["Southeast Asia"]["countries_included"],
            # Technical Specifications
            "framework": "ONNX",
            "input_shape": 15,
            "input_channels": [
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B11",
                "B12",
                "NDVI",
                "VV",
                "VH",
                "DEM",
                "lon",
                "lat",
            ],
            "output_shape": 24,
            "output_classes": [
                "Background",
                "Other_large_scale_cropland",
                "Pasture",
                "Mining",
                "Other_small_scale_cropland",
                "Roads",
                "Forest",
                "Plantation_forest",
                "Coffee",
                "Build_up",
                "Water",
                "Oil_palm",
                "Rubber",
                "Cacao",
                "Avocado",
                "Soy",
                "Sugar",
                "Maize",
                "Banana",
                "Pineapple",
                "Rice",
                "Wood_logging",
                "Cashew",
                "Tea",
            ],
            # Temporal Information
            "temporal_start": "2020-01-01",
            "temporal_end": "2030-12-31",
            # Storage Information
            "model_url": "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/WorldAgriCommodities/models/best_weights_att_unet_lagtime_5_Fused3_2023_totalLoss6V1_without_loss_sent_Southeast_Asia20.onnx",
            # Performance Metrics
            "overall_accuracy": "None",
            "f1_score_macro": "None",
        }
    )

    # Latin America Model (updated to match your structure)
    models_data.append(
        {
            # Basic Identification
            "model_id": "WorldAgriCommodities_LatinAmerica_v1",
            "model_name": "Latin America Crop Classification 2024",
            "region": "Latin America",
            "description": "Deep learning model for crop classification across Latin America",
            # Spatial Information
            "bbox": str(REGION_BBOXES["Latin America"]["bbox"]),
            "bbox_min_lon": REGION_BBOXES["Latin America"]["bbox"][0],
            "bbox_min_lat": REGION_BBOXES["Latin America"]["bbox"][1],
            "bbox_max_lon": REGION_BBOXES["Latin America"]["bbox"][2],
            "bbox_max_lat": REGION_BBOXES["Latin America"]["bbox"][3],
            "crs": "EPSG:4326",
            "countries_covered": REGION_BBOXES["Latin America"]["countries_included"],
            # Technical Specifications
            "framework": "ONNX",
            "input_shape": 15,
            "input_channels": [
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B11",
                "B12",
                "NDVI",
                "VV",
                "VH",
                "DEM",
                "lon",
                "lat",
            ],
            "output_shape": 22,
            "output_classes": [
                "Background",
                "Other_large_scale_cropland",
                "Pasture",
                "Mining",
                "Other_small_scale_cropland",
                "Roads",
                "Forest",
                "Plantation_forest",
                "Coffee",
                "Build_up",
                "Water",
                "Oil_palm",
                "Rubber",
                "Cacao",
                "Avocado",
                "Soy",
                "Sugar",
                "Maize",
                "Banana",
                "Pineapple",
                "Rice",
                "Wood_logging",
            ],
            # Temporal Information
            "temporal_start": "2020-01-01",
            "temporal_end": "2030-12-31",
            # Storage Information
            "model_url": "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/WorldAgriCommodities/models/best_weights_att_unet_lagtime_5_Fused3_2023_totalLoss6V1_without_loss_sent_Latin_America53.onnx",
            # Performance Metrics
            "overall_accuracy": "None",
            "f1_score_macro": "None",
        }
    )

    return pd.DataFrame(models_data)


# Save the enhanced metadata
def save_enhanced_metadata(df_models, output_dir="model_metadata"):
    """Save the enhanced metadata in multiple formats"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save as CSV
    csv_path = output_dir / "model_metadata.csv"

    # Convert list columns to strings for CSV
    df_export = df_models.copy()
    df_export["input_channels"] = df_export["input_channels"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )
    df_export["output_classes"] = df_export["output_classes"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )
    df_export["countries_covered"] = df_export["countries_covered"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )

    df_export.to_csv(csv_path, index=False)
    print(f"\nEnhanced metadata saved to: {csv_path}")

    return csv_path


# Save the metadata
df_models_enhanced = create_enhanced_model_metadata()
csv_path = save_enhanced_metadata(df_models_enhanced)
