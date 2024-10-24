import json
json_path = "examples/mp/ffbf0023_p0.json"

with open(json_path, "r", encoding='utf-8') as f:
    data = json.load(f)

with open(json_path,'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
