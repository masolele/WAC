
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
    """Class that handles authentication for the VITO STAC API."""
    
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
        """Get API bearer access token via password flow."""
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

def load_model_metadata(csv_path):
    """Load and parse the model metadata CSV"""
    df = pd.read_csv(csv_path)
    
    # Convert string representations back to lists
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
    """Create a geometry from bbox coordinates"""
    return box(bbox[0], bbox[1], bbox[2], bbox[3])

def create_stac_items(df_models, collection_id):
    """Create STAC items for each model"""
    
    items = []
    
    for _, model in df_models.iterrows():
        model_id = model['model_id']
        print(f"Creating STAC item: {model_id}")
        
        # Create geometry from bbox
        bbox = model['bbox']
        geometry = create_geometry_from_bbox(bbox)
        
        # Parse temporal extent
        start_date = datetime.strptime(model['temporal_start'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(model['temporal_end'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        # Create properties - ensure all values are JSON serializable
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
        
        # Create STAC Item
        item = Item(
            id=str(model_id),
            geometry=geometry.__geo_interface__,
            bbox=[float(x) for x in bbox],
            datetime=None,
            start_datetime=start_date,
            end_datetime=end_date,
            properties=properties
        )
        
        # Add model asset
        model_asset = Asset(
            href=str(model['model_url']),
            media_type="application/onnx",
            title=f"ONNX Model - {model['model_name']}",
            description=str(model['description']),
            roles=["ml-model", "inference"]
        )
        item.add_asset('model', model_asset)
        
        # Set collection ID
        item.collection_id = collection_id
        
        items.append(item)
        print(f" Created STAC item: {model_id}")
    
    return items

def item_exists(catalog_url, collection_id, item_id, auth):
    """Check if item already exists"""
    try:
        response = requests.get(
            f"{catalog_url}/collections/{collection_id}/items/{item_id}", 
            auth=auth
        )
        return response.status_code == 200
    except:
        return False

def upload_items_simple(catalog_url, collection_id, items, auth):
    """Upload items - handles both new items and updates"""
    items_url = f"{catalog_url}/collections/{collection_id}/items"
    
    successful = 0
    failed = 0
    skipped = 0
    
    print(f"\n Uploading {len(items)} items to collection '{collection_id}'...")
    
    for item in items:
        try:
            # Check if item already exists
            if item_exists(catalog_url, collection_id, item.id, auth):
                print(f"  {item.id} already exists - skipping")
                skipped += 1
                continue
            
            # Simple upload like your working example
            response = requests.post(
                url=items_url,
                json=item.to_dict(),
                auth=auth
            )
            
            if response.status_code in [200, 201]:
                print(f"  {item.id} uploaded successfully")
                successful += 1
            else:
                print(f"  {item.id} failed: {response.status_code} - {response.text}")
                failed += 1
                
        except Exception as e:
            print(f"  {item.id} error: {e}")
            failed += 1
    
    print(f"\n Upload summary:")
    print(f"   Successful: {successful}")
    print(f"   Skipped (already exist): {skipped}")
    print(f"   Failed: {failed}")
    
    return successful, skipped, failed

def save_stac_locally(collection, items, output_dir):
    """Save STAC locally for inspection"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save collection
    collection_path = output_path / "collection.json"
    with open(collection_path, 'w') as f:
        import json
        json.dump(collection.to_dict(), f, indent=2)
    print(f"Collection saved: {collection_path}")
    
    # Save items
    items_dir = output_path / "items"
    items_dir.mkdir(exist_ok=True)
    
    for item in items:
        item_path = items_dir / f"{item.id}.json"
        with open(item_path, 'w') as f:
            json.dump(item.to_dict(), f, indent=2)
        print(f"Item saved: {item_path}")
    
    return output_path

def main():
    """Main function - focused on uploading items to existing collection"""
    
    print("Building STAC items from CSV...")
    
    try:
        # Load model metadata
        df_models = load_model_metadata(config["metadata_csv"])
        
        # Create STAC items
        items = create_stac_items(df_models, config["collection_name"])
        
        if not items:
            print("No valid items could be created")
            return
        
        # Create a minimal collection object for local saving (not for upload)
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
        
        # Save locally first
        print(f"\nSaving STAC locally...")
        output_path = save_stac_locally(collection, items, config["output_directory"])
        
        print(f"\nLocal STAC creation completed!")
        print(f"   Output directory: {output_path}")
        print(f"   Collection: {config['collection_name']}")
        print(f"   Items: {len(items)}")
        
        # Upload to STAC API

        print(f"\n Uploading to STAC API...")
        password = getpass.getpass("Enter password: ")
        config["password"] = password
        
        # Initialize authentication
        auth = VitoStacApiAuthentication(
            username=config["username"],
            password=config["password"]
        )
        
        # Upload items to existing collection
        successful, skipped, failed = upload_items_simple(
            config["catalog_url"], 
            config["collection_name"], 
            items, 
            auth
        )
        
        if failed == 0:
            if successful > 0:
                print(f"\n STAC item upload completed successfully!")
                print(f"   {successful} new items added to collection '{config['collection_name']}'")
            if skipped > 0:
                print(f"   {skipped} items already exist and were skipped")
        else:
            print(f"\n STAC item upload partially completed")
            print(f"   {successful} successful, {skipped} skipped, {failed} failed")
                
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()