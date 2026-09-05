import json

def fix_input_layer_configs(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") == "InputLayer" and "config" in obj:
            cfg = obj["config"]
            if "batch_shape" in cfg:
                cfg["batch_input_shape"] = cfg.pop("batch_shape")
            cfg.pop("optional", None)
        for v in obj.values():
            fix_input_layer_configs(v)
    elif isinstance(obj, list):
        for item in obj:
            fix_input_layer_configs(item)

with open("emotiondetector.json", "r") as f:
    model_config = json.load(f)

fix_input_layer_configs(model_config)

with open("emotiondetector.json", "w") as f:
    json.dump(model_config, f)

print("Fixed emotiondetector.json")