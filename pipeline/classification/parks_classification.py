import ee
import geopandas as gpd
import folium
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pattern_classifier import predict_image_probabilities
from folium.plugins import HeatMap
from PIL import Image
from io import BytesIO
import requests

# --- Konfiguracja ---
PARKS_SHAPEFILE_PATH = "../../data/italy_nationalparks_geodata/natural.shp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Preparation ---
# Load pre-trained ResNet50 and remove final layer for feature extraction
def prepare_model():
    # Feature extractor
    feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    feature_extractor.fc = nn.Identity()
    feature_extractor = feature_extractor.to(DEVICE)
    feature_extractor.eval()
    # Dataset to infer class names
    data_path = '../../data/classification'
    dataset = datasets.ImageFolder(data_path, transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor()
    ]))
    class_names = dataset.classes
    # Extract features for classifier training
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    features, labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)
            feats = feature_extractor(imgs).cpu().numpy()
            features.extend(feats)
            labels.extend(labs.numpy())
    # Train logistic regression classifier
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    classifier = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
    classifier.fit(X_train, y_train)
    return feature_extractor, classifier, class_names

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='italy-desertification')
# Prepare model once
feature_extractor_model, classifier_model, class_names = prepare_model()

# --- Main classification function ---
def classify_park(ee_geom, park_name, ntiles=50):
    # Sentinel-2 composite
    s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filterDate('2021-06-01', '2021-08-31') \
        .filterBounds(ee_geom) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median() \
        .select(['B4','B3','B2'])

    bounds = ee_geom.bounds().getInfo()['coordinates'][0]
    lons = [pt[0] for pt in bounds]
    lats = [pt[1] for pt in bounds]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    results = []
    for i in range(ntiles):
        lon = lon_min + (lon_max - lon_min) * torch.rand(1).item()
        lat = lat_min + (lat_max - lat_min) * torch.rand(1).item()
        pt = ee.Geometry.Point([lon, lat])
        thumb_url = s2.getThumbURL({
            'region': pt.buffer(1120).bounds().getInfo(),
            'dimensions': [224, 224], 'format': 'jpg', 'min': 0, 'max': 3000
        })
        try:
            resp = requests.get(thumb_url)
            img = Image.open(BytesIO(resp.content)).convert('RGB')
        except:
            continue
        img_path = f"/vsimem/{park_name}_{i}.jpg"
        img.save(img_path)
        probs = predict_image_probabilities(
            img_path, feature_extractor_model, classifier_model, class_names
        )
        results.append((lat, lon, probs))
    return results

# --- Script entry point ---
def main():
    parks_gdf = gpd.read_file(PARKS_SHAPEFILE_PATH).to_crs(epsg=4326)
    m = folium.Map(location=[42.5,12.5], zoom_start=6, tiles="CartoDB positron")
    heat_data = {cls: [] for cls in class_names}

    for idx, row in parks_gdf.iterrows():
        name = row['NAME']
        ee_geom = ee.Geometry(row['geometry'].__geo_interface__)
        points = classify_park(ee_geom, name)
        for lat, lon, probs in points:
            for cls in class_names:
                heat_data[cls].append([lat, lon, probs.get(cls, 0)])

    for cls, data in heat_data.items():
        HeatMap(data, name=f"Heatmap: {cls}", radius=15, blur=10, max_zoom=6).add_to(m)
    folium.LayerControl().add_to(m)
    out = "italian_parks_classification_heatmap.html"
    m.save(out)
    print(f"Mapa zapisana jako {out}")

if __name__ == '__main__':
    main()
