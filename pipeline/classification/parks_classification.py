import os
import ee
import geopandas as gpd
import folium
import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pattern_classifier import predict_image_probabilities
from folium.plugins import HeatMap
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import pyogrio

# --- Ustawienia środowiska dla OGR ---
os.environ['OGR_GEOMETRY_ACCEPT_UNCLOSED_RING'] = 'YES'

# --- Konfiguracja ---
PARKS_SHAPEFILE_PATH = "../../data/italy_national_parks_geodata/italy_polygon.shp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Preparation ---
def prepare_model():
    feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    feature_extractor.fc = nn.Identity()
    feature_extractor = feature_extractor.to(DEVICE)
    feature_extractor.eval()

    data_path = '../../data/classification'
    dataset = datasets.ImageFolder(
        data_path,
        transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    )
    class_names = dataset.classes

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    features, labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)
            feats = feature_extractor(imgs).cpu().numpy()
            features.extend(feats)
            labels.extend(labs.numpy())
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=0.25, random_state=42, stratify=labels
    )
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)
    return feature_extractor, classifier, class_names

# --- Inicjalizacja Earth Engine ---
os.environ.setdefault('EARTHENGINE_PROJECT', 'YOUR_PROJECT_ID')
ee.Authenticate()
ee.Initialize(project='italy-desertification')
feature_extractor_model, classifier_model, class_names = prepare_model()

# --- Funkcja klasyfikacji ---
def classify_park(ee_geom, park_name, ntiles=50):
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
        .filterDate('2010-06-01', '2025-08-31')
        .filterBounds(ee_geom)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )
    # Sprawdź, czy cokolwiek się znalazło
    count = collection.size().getInfo()
    if count == 0:
        print(f"Brak obrazów dla parku {park_name}, pomijam...")
        return []
    else:
        print(f"Znaleziono obrazy dla parku {park_name},")
    # Tworzymy medianę i wybieramy kanały
    s2 = collection.median().select(['B4', 'B3', 'B2'])
    coords = ee_geom.bounds().getInfo()['coordinates'][0]
    lons, lats = zip(*coords)
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    results = []
    for i in range(ntiles):
        lon = lon_min + (lon_max-lon_min)*torch.rand(1).item()
        lat = lat_min + (lat_max-lat_min)*torch.rand(1).item()
        pt = ee.Geometry.Point([lon, lat])
        thumb_url = s2.getThumbURL({
            'region': pt.buffer(1120).bounds().getInfo(),
            'dimensions': [224, 224], 'format': 'jpg', 'min':0, 'max':3000
        })
        try:
            resp = requests.get(thumb_url)
            img = Image.open(BytesIO(resp.content)).convert('RGB')
        except:
            continue
        img_path = f"/vsimem/{park_name}_{i}.jpg"
        img.save(img_path)
        probs = predict_image_probabilities(img_path, feature_extractor_model, classifier_model, class_names)
        results.append((lat, lon, probs))
    return results


def main():
    parks_gdf = pyogrio.read_dataframe(
        PARKS_SHAPEFILE_PATH,
        on_invalid='fix'
    )
    parks_gdf = parks_gdf[parks_gdf["type"] == "park"]
    parks_gdf.plot(edgecolor='black', facecolor='lightgreen')
    plt.title("Wizualizacja parków narodowych")
    plt.axis('equal')
    plt.show()

    m = folium.Map(location=[42.5, 12.5], zoom_start=6, tiles="CartoDB positron")
    heat_data = {cls: [] for cls in class_names}

    for _, row in parks_gdf.iterrows():
        ee_geom = ee.Geometry(row['geometry'].__geo_interface__)
        points = classify_park(ee_geom, row.get('NAME', 'park'))
        for lat, lon, probs in points:
            for cls in class_names:
                heat_data[cls].append([lat, lon, probs.get(cls, 0)])

    for cls, data in heat_data.items():
        HeatMap(data, name=f"Heatmap: {cls}", radius=15, blur=10).add_to(m)
    folium.LayerControl().add_to(m)
    out_file = "italian_parks_classification_heatmap.html"
    m.save(out_file)
    print(f"Mapa zapisana jako {out_file}")


if __name__ == '__main__':
    ntiles = 10
    parks_gdf = gpd.read_file(PARKS_SHAPEFILE_PATH, on_invalid='fix')
    print("No. of records: ", parks_gdf.shape)
    print("CRS shapefile:", parks_gdf.crs)
    print("Kolumny:", parks_gdf.columns.tolist())
    print(parks_gdf.head()[['geometry'] + parks_gdf.columns.tolist()[:3]])
    parks_gdf.plot(edgecolor='black', facecolor='lightgreen')
    plt.title("Wizualizacja parków narodowych")
    plt.axis('equal')
    plt.show()
    m = folium.Map(location=[42.5, 12.5], zoom_start=6, tiles="CartoDB positron")
    heat_data = {cls: [] for cls in class_names}
    for _, row in tqdm.tqdm(parks_gdf.iterrows(), desc='Park steps'):
        ee_geom = ee.Geometry(row['geometry'].__geo_interface__)
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterDate('2010-06-01', '2025-08-31')
            .filterBounds(ee_geom)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        )
        # Sprawdź, czy cokolwiek się znalazło
        count = collection.size().getInfo()
        if count == 0:
            print(f"Brak obrazów dla parku, pomijam...")
        else:
            print(f"Znaleziono obrazy dla parku,")

        s2 = collection.select(['B4', 'B3', 'B2']).median()
        coords = ee_geom.bounds().getInfo()['coordinates'][0]
        lons, lats = zip(*coords)
        lon_min, lon_max = min(lons), max(lons)
        lat_min, lat_max = min(lats), max(lats)
        print("Wspolrzedne: ", (lon_min, lon_max, lat_min, lat_max))
        points = []
        for i in tqdm.tqdm(range(ntiles)):
            lon = lon_min + (lon_max - lon_min) * torch.rand(1).item()
            lat = lat_min + (lat_max - lat_min) * torch.rand(1).item()
            pt = ee.Geometry.Point([lon, lat])
            thumb_url = s2.getThumbURL({
                'region': pt.buffer(1120).bounds().getInfo(),
                'dimensions': [224, 224], 'format': 'jpg', 'min': 0, 'max': 3000
            })
            print("Thumb URL:", thumb_url)
            try:
                resp = requests.get(thumb_url)
                img = Image.open(BytesIO(resp.content)).convert('RGB')
            except Exception as error:
                print("Failed to download image: ", error)
                continue
            img_path = f"../../data/italy_national_parks_screenshots_ee/park_{i}.jpg"
            img.save(img_path)
            probs = predict_image_probabilities(img_path, feature_extractor_model, classifier_model, class_names)
            print("Estimated probs: ", probs)
            points.append((lat, lon, probs))
            for lat, lon, probs in points:
                for cls in class_names:
                    heat_data[cls].append([lat, lon, probs.get(cls, 0)])
    for cls, data in heat_data.items():
        HeatMap(data, name=f"Heatmap: {cls}", radius=15, blur=10).add_to(m)
    folium.LayerControl().add_to(m)
    out_file = "italian_parks_classification_heatmap.html"
    m.save(out_file)
