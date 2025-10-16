
#%%

import requests
import getpass
from pathlib import Path


import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from shapely.geometry import box
import requests
from requests.auth import AuthBase

# STAC imports
import pystac
from pystac import Item, Collection, Extent, SpatialExtent, TemporalExtent, Asset, Link

from openeo.rest.auth.oidc import (
    OidcClientInfo,
    OidcProviderInfo,
    OidcResourceOwnerPasswordAuthenticator,
)

# Configuration
config = {
    "metadata_csv": "model_metadata/model_metadata.csv",
    "collection_name": "world-agri-commodities-models",
    "output_directory": "stac_output",
    "catalog_url": "https://stac.openeo.vito.be", 
    "username": "hans.vanrompay@vito.be",  # Terrascope username
}

class VitoStacApiAuthentication(AuthBase):
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self._access_token = None

    def __call__(self, request):
        if self._access_token is None:
            self._access_token = self.get_access_token()
        request.headers["Authorization"] = self._access_token
        return request

    def get_access_token(self) -> str:
        provider_info = OidcProviderInfo(
            issuer="https://sso.terrascope.be/auth/realms/terrascope"
        )
        client_info = OidcClientInfo(
            client_id="terracatalogueclient",
            provider=provider_info,
        )
        if self.username and self.password:
            authenticator = OidcResourceOwnerPasswordAuthenticator(
                client_info=client_info, username=self.username, password=self.password
            )
            tokens = authenticator.get_tokens()
            return f"Bearer {tokens.access_token}"
        else:
            raise ValueError("Credentials are required to obtain an access token.")

def collection_exists(catalog_url, collection_id, auth):
    """Check if collection exists"""
    try:
        response = requests.get(f"{catalog_url}/collections/{collection_id}", auth=auth)
        return response.status_code == 200
    except:
        return False

def create_collection_with_auth(catalog_url, collection, auth):
    """Create collection with proper authorization structure like your colleague's example"""
    coll_dict = collection.to_dict()
    
    # Clear links that might cause issues
    if "links" in coll_dict:
        coll_dict["links"] = []
    
    # Add the _auth field that your colleague's working code uses
    coll_dict.setdefault("_auth", {"read": ["anonymous"], "write": ["stac-admin-prod"]})
    
    print("Attempting to create collection...")
    response = requests.post(
        f"{catalog_url}/collections",
        auth=auth,
        json=coll_dict
    )
    
    if response.status_code in [200, 201]:
        print(f"Collection created: {collection.id}")
        return True
    else:
        print(f"Failed to create collection: {response.status_code} - {response.text}")
        return False

def load_model_metadata(csv_path):
    df = pd.read_csv(csv_path)
    
    if 'input_channels' in df.columns:
        df['input_channels'] = df['input_channels'].apply(lambda x: [item.strip() for item in x.split(',')] if pd.notna(x) else [])
    if 'output_classes' in df.columns:
        df['output_classes'] = df['output_classes'].apply(lambda x: [item.strip() for item in x.split(',')] if pd.notna(x) else [])
    if 'countries_covered' in df.columns:
        df['countries_covered'] = df['countries_covered'].apply(lambda x: [item.strip() for item in x.split(',')] if pd.notna(x) else [])
    if 'bbox' in df.columns:
        df['bbox'] = df['bbox'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')] if pd.notna(x) else [])
    
    print(f"Loaded {len(df)} models from {csv_path}")
    return df

def create_geometry_from_bbox(bbox):
    return box(bbox[0], bbox[1], bbox[2], bbox[3])

def create_stac_items(df_models, collection_id):
    items = []
    
    for _, model in df_models.iterrows():
        model_id = model['model_id']
        print(f"Creating STAC item: {model_id}")
        
        bbox = model['bbox']
        geometry = create_geometry_from_bbox(bbox)
        
        start_date = datetime.strptime(model['temporal_start'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(model['temporal_end'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        properties = {
            "modelID": str(model['model_id']),
            "name": str(model['model_name']),
            "region": str(model['region']),
            "description": str(model['description']),
            "countries_covered": [str(x) for x in model.get('countries_covered', [])],
            "framework": str(model.get('framework', 'ONNX')),
            "input_shape": int(model['input_shape']),
            "output_shape": int(model['output_shape']),
            "input_channels": [str(x) for x in model.get('input_channels', [])],
            "output_classes": [str(x) for x in model.get('output_classes', [])],
            "number_input_features": int(len(model.get('input_channels', []))),
            "number_output_classes": int(len(model.get('output_classes', []))),
            "model_urls": [str(model['model_url'])],
            "temporal_extent": [start_date.isoformat(), end_date.isoformat()],
            "overall_accuracy": str(model.get('overall_accuracy', 'None')),
            "f1_score_macro": str(model.get('f1_score_macro', 'None')),
        }
        
        item = Item(
            id=str(model_id),
            geometry=geometry.__geo_interface__,
            bbox=[float(x) for x in bbox],
            datetime=None,
            start_datetime=start_date,
            end_datetime=end_date,
            properties=properties
        )
        
        model_asset = Asset(
            href=str(model['model_url']),
            media_type="application/onnx",
            title=f"ONNX Model - {model['model_name']}",
            description=str(model['description']),
            roles=["ml-model", "inference"]
        )
        item.add_asset('model', model_asset)
        
        item.collection_id = collection_id
        items.append(item)
        print(f"Created STAC item: {model_id}")
    
    return items

def upload_items_simple(catalog_url, collection_id, items, auth):
    items_url = f"{catalog_url}/collections/{collection_id}/items"
    
    successful = 0
    failed = 0
    
    print(f"Uploading {len(items)} items to collection '{collection_id}'...")
    
    for item in items:
        try:
            # Clear links for items too, like your colleague's example
            item_dict = item.to_dict()
            if "links" in item_dict:
                item_dict["links"] = []
                
            response = requests.post(
                url=items_url,
                json=item_dict,
                auth=auth
            )
            
            if response.status_code in [200, 201]:
                print(f"{item.id} uploaded successfully")
                successful += 1
            else:
                print(f"{item.id} failed: {response.status_code} - {response.text}")
                failed += 1
                
        except Exception as e:
            print(f"{item.id} error: {e}")
            failed += 1
    
    print(f"Upload summary: {successful} successful, {failed} failed")
    return successful, failed

def save_stac_locally(collection, items, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    collection_path = output_path / "collection.json"
    with open(collection_path, 'w') as f:
        import json
        json.dump(collection.to_dict(), f, indent=2)
    print(f"Collection saved: {collection_path}")
    
    items_dir = output_path / "items"
    items_dir.mkdir(exist_ok=True)
    
    for item in items:
        item_path = items_dir / f"{item.id}.json"
        with open(item_path, 'w') as f:
            json.dump(item.to_dict(), f, indent=2)
        print(f"Item saved: {item_path}")
    
    return output_path

def main():
    print("Building STAC items from CSV...")
    
    try:
        df_models = load_model_metadata(config["metadata_csv"])
        items = create_stac_items(df_models, config["collection_name"])
        
        if not items:
            print("No valid items could be created")
            return
        
        # Create collection object
        collection = Collection(
            id=config["collection_name"],
            title="World Agricultural Commodities Classification Models",
            description="STAC collection containing deep learning models for crop classification",
            extent=Extent(
                spatial=SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
                temporal=TemporalExtent([[
                    datetime(2020, 1, 1, tzinfo=timezone.utc),
                    datetime(2030, 12, 31, tzinfo=timezone.utc)
                ]])
            ),
            license="proprietary"
        )
        
        print("Saving STAC locally...")
        output_path = save_stac_locally(collection, items, config["output_directory"])
        
        print("Local STAC creation completed!")
        print(f"Output directory: {output_path}")
        print(f"Collection: {config['collection_name']}")
        print(f"Items: {len(items)}")
        
        print("Uploading to STAC API...")
        password = getpass.getpass("Enter password: ")
        
        auth = VitoStacApiAuthentication(
            username=config["username"],
            password=password
        )
        
        # Check if collection exists
        if not collection_exists(config["catalog_url"], config["collection_name"], auth):
            print(f"Collection '{config['collection_name']}' does not exist. Creating it...")
            if create_collection_with_auth(config["catalog_url"], collection, auth):
                print("Collection created successfully")
            else:
                print("Failed to create collection. Cannot upload items.")
                return
        else:
            print(f"Collection '{config['collection_name']}' already exists")
        
        # Upload items
        successful, failed = upload_items_simple(
            config["catalog_url"], 
            config["collection_name"], 
            items, 
            auth
        )
        
        if failed == 0:
            print(f"STAC deployment completed successfully! {successful} items uploaded.")
        else:
            print(f"STAC deployment partially completed: {successful} successful, {failed} failed")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()