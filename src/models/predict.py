import sys
import os
import json
import joblib
import numpy as np

backend_path = '/workspaces/airbnb-price-prediction/backend'
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

print("=" * 60)
print("SEARCHING ENTIRE REPO FOR MODELS...")
print("=" * 60)
print()

# Search for models
found_models = {}
root_dir = '/workspaces/airbnb-price-prediction'

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.pkl') or filename.endswith('.joblib'):
            full_path = os.path.join(dirpath, filename)
            try:
                model = joblib.load(full_path)
                model_name = filename.replace('.pkl', '').replace('.joblib', '')
                found_models[model_name] = model
                print(f"✅ FOUND: {model_name}")
                print(f"   Location: {full_path}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

print()
print(f"Total models found: {len(found_models)}")
print()

if len(found_models) == 0:
    print("ERROR: No models found!")
    sys.exit(1)

tests = [
    {"name": "Test 1: Standard Listing", "data": [0, 2, 1, 4, 1, 5, 1, 0.75, 1, 4.5]},
    {"name": "Test 2: Premium - Marais", "data": [0, 3, 2, 6, 1, 8, 1, 0.90, 1, 4.8]},
    {"name": "Test 3: Private Room", "data": [1, 1, 1, 2, 1, 12, 0, 0.60, 0, 4.0]},
    {"name": "Test 4: Shared Room", "data": [2, 1, 1, 1, 1, 20, 0, 0.30, 0, 3.5]},
    {"name": "Test 5: Luxury", "data": [0, 4, 3, 8, 1, 2, 1, 0.95, 1, 4.9]}
]

output_results = {
    "status": "SUCCESS",
    "models_found": len(found_models),
    "model_list": list(found_models.keys()),
    "predictions": []
}

print("=" * 60)
print("RUNNING PREDICTIONS (RAW 10-FEATURE INPUT)")
print("=" * 60)
print()

for t in tests:
    # Use raw 10 features directly - NO transformation
    X = np.array(t["data"]).reshape(1, -1)
    
    pred_result = {
        "test_name": t["name"],
        "input_data": t["data"],
        "predictions": {}
    }
    
    predictions_list = []
    
    for model_name, model in found_models.items():
        try:
            # Handle pipeline dict format
            if isinstance(model, dict) and 'pipeline' in model:
                pred = float(np.expm1(model['pipeline'].predict(X)[0]))
            else:
                pred = float(np.expm1(model.predict(X)[0]))
            
            pred_result["predictions"][model_name] = round(pred, 2)
            predictions_list.append(pred)
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    if predictions_list:
        avg = sum(predictions_list) / len(predictions_list)
        pred_result["average_price_eur"] = round(avg, 2)
        output_results["predictions"].append(pred_result)
        
        print(f"{t['name']}")
        for model_name, price in pred_result["predictions"].items():
            print(f"  {model_name}: €{price:.2f}")
        print(f"  AVERAGE: €{pred_result['average_price_eur']:.2f}\n")

json_path = "/workspaces/airbnb-price-prediction/results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_results, f, indent=4)

print("=" * 60)
print(f"✅ TESTS COMPLETED")
print(f"✅ {len(found_models)} MODELS USED")
print(f"✅ RESULTS SAVED TO {json_path}")
print("=" * 60)
