import json
with open('data/final/phase3_roi_analysis.json') as f:
    data = json.load(f)

for key in ['per_layer_faithful_vs_override', 'per_layer_category_comparison']:
    print(f"=== {key} ===")
    val = data[key]
    print("type:", type(val))
    if isinstance(val, dict):
        print("keys:", list(val.keys()))
        # peek one level deeper into the first key
        first_k = list(val.keys())[0]
        inner = val[first_k]
        print(f"  ['{first_k}'] type:", type(inner))
        if isinstance(inner, list):
            print(f"  ['{first_k}'][0]:", json.dumps(inner[0], indent=2))
        elif isinstance(inner, dict):
            print(f"  ['{first_k}'] keys:", list(inner.keys()))
    print()
