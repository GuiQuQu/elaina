import json
json_path = "jlxn0226_p6.json"

with open(json_path, "r", encoding='utf-8') as f:
    data = json.load(f)

with open(json_path,'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
