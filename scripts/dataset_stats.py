import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'LogDLR'))

from utils.data import LogDataModule
from sentence_transformers import SentenceTransformer
import torch

DATASETS = {
    'Thunderbird': '/Users/anurag/BTP/LogLDR+++/LogDLR/dataset/Thunderbird/Thunderbird.log',
    'Spirit': '/Users/anurag/BTP/LogLDR+++/LogDLR/dataset/Spirit/Spirit.log',
    'BGL': '/Users/anurag/BTP/LogLDR+++/LogDLR/dataset/BGL/BGL.log',
}

NROWS = 10_000_000
WINDOW_SIZE = 20
STEP_SIZE = 4


def count_logs_and_windows(log_file, nrows, window_size, step_size):
    normal_logs, anomaly_logs = 0, 0
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for i, row in enumerate(f):
            if i >= nrows:
                break
            fields = row.strip().split(maxsplit=9)
            if len(fields) == 10:
                if fields[0] != '-':
                    anomaly_logs += 1
                else:
                    normal_logs += 1
    total_logs = normal_logs + anomaly_logs
    total_windows = max(0, (total_logs - window_size) // step_size + 1)
    # approximate; exact sliding window count with step
    return {
        'total_log_lines_read': total_logs,
        'normal_log_lines': normal_logs,
        'anomaly_log_lines': anomaly_logs,
        'approx_normal_windows': max(0, total_windows - (anomaly_logs // step_size)),
        'approx_anomaly_windows': anomaly_logs // step_size,
    }


def main():
    stats = {}
    for name, path in DATASETS.items():
        print(f'Computing statistics for {name} ...')
        stats[name] = count_logs_and_windows(path, NROWS, WINDOW_SIZE, STEP_SIZE)
        stats[name]['log_file'] = path
        print(json.dumps({name: stats[name]}, indent=2))

    out_path = Path('/Users/anurag/BTP/LogLDR+++/results/dataset_statistics.json')
    out_path.write_text(json.dumps(stats, indent=2))
    print(f'\nSaved dataset statistics to {out_path}')


if __name__ == '__main__':
    main()
