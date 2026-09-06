import json
from collections import Counter

TITLES_FILE = "data/final/dataset_1500_with_titles_1.jsonl"

records = []
with open(TITLES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

titles = [r.get("gold_title") or r.get("title") for r in records]
populated = [t for t in titles if t]
unique = set(populated)

print(f"Rows with a populated title: {len(populated)}")
print(f"Unique titles among those: {len(unique)}")
print(f"Difference (duplicate title strings across rows): {len(populated) - len(unique)}")

counts = Counter(populated)
dupes = {t: c for t, c in counts.items() if c > 1}
print(f"\nTitles shared by 2+ rows: {len(dupes)}")
for t, c in sorted(dupes.items(), key=lambda x: -x[1])[:10]:
    print(f"  {c}x  {t!r}")