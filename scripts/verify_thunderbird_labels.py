"""
Independent verification of Thunderbird label counts using the exact parsing
logic in LogDLR/utils/data.py.
"""
import json
from pathlib import Path

LOG_FILE = Path('/Users/anurag/BTP/LogLDR+++/LogDLR/dataset/Thunderbird/Thunderbird.log')
NUM_FIELDS = 10
NROWS = 10_000_000

labels = []
with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    for i, row in enumerate(f):
        if i >= NROWS:
            break
        fields = row.strip().split(maxsplit=NUM_FIELDS - 1)
        if len(fields) == NUM_FIELDS:
            labels.append(fields[0] != '-')

normal_lines = sum(1 for x in labels if not x)
anomaly_lines = sum(1 for x in labels if x)

# First-field distribution
from collections import Counter
distribution = Counter()
with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    for i, row in enumerate(f):
        if i >= NROWS:
            break
        fields = row.strip().split(maxsplit=NUM_FIELDS - 1)
        if len(fields) == NUM_FIELDS:
            distribution[fields[0]] += 1

result = {
    'file': str(LOG_FILE),
    'nrows_read': len(labels),
    'normal_lines': normal_lines,
    'anomaly_lines': anomaly_lines,
    'first_field_distribution': dict(distribution.most_common()),
    'note': 'This uses the exact label convention of LogDLR/utils/data.py: anomaly iff first field != "-".'
}

out = Path('/Users/anurag/BTP/LogLDR+++/results/thunderbird_label_verification.json')
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
