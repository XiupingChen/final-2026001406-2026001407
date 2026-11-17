"""Validate completed Windows artifacts and write a concise measured report."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def markdown_table(frame):
    headers = list(frame.columns)
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in row) + ' |')
    return '\n'.join(lines)


def verify(directory):
    root = Path(directory).resolve()
    checks = []
    expected = {'smoke': 8, 'phase1_prediction': 12, 'phase2_parallel': 40, 'phase3_scaling': 120}
    for stage, count in expected.items():
        path = root / stage
        complete = json.loads((path / 'COMPLETE.json').read_text(encoding='utf-8'))
        consistency = pd.read_csv(path / 'consistency.csv')
        timings = pd.read_csv(path / 'timings.csv')
        assert complete['jobs'] == len(consistency) == len(timings) == count
        assert consistency.passed.all() and consistency.max_absolute_error.max() <= 1e-12
        assert (timings.core_seconds > 0).all() and (timings.process_wall_seconds >= timings.core_seconds).all()
        manifest = json.loads((path / 'prepared/manifest.json').read_text(encoding='utf-8'))
        for rel, hash_value in manifest['sha256'].items():
            assert digest(path / 'prepared' / rel.replace('\\', '/')) == hash_value, (stage, rel)
        jobs = list((path / 'raw').glob('*/predictions.npz'))
        assert len(jobs) == count
        maximum = 0.
        for file in jobs:
            name = file.parent.name.split('__')
            reference = path / 'raw' / f'{name[0]}__{name[1]}__serial__r0' / 'predictions.npz'
            with np.load(file) as current, np.load(reference) as baseline:
                a, b = current['probabilities'], baseline['probabilities']
                assert np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
                np.testing.assert_array_equal(current['evaluation_row_ids'], baseline['evaluation_row_ids'])
                maximum = max(maximum, float(np.max(np.abs(a-b))))
        assert maximum <= 1e-12
        checks.append({'stage': stage, 'jobs': count, 'recomputed_max_schedule_error': maximum,
                       'prepared_hashes_verified': len(manifest['sha256'])})
    # Verify B=40 extends B=20 without changing earlier draws/model seeds.
    prepared = root / 'phase3_scaling/prepared'
    for filename in ['bootstrap.csv', 'seeds.csv']:
        a = np.loadtxt(prepared / 'base' / filename, delimiter=',')
        b = np.loadtxt(prepared / 'B_40' / filename, delimiter=',')
        np.testing.assert_array_equal(a, b[:20])
    sensitivity = pd.read_csv(root / 'sensitivity/metrics.csv')
    audits = pd.read_csv(root / 'sensitivity/split_audit.csv')
    assert len(sensitivity) == 12
    assert audits.query("split == 'grouped'").test_rows_matching_train_predictors.sum() == 0
    for split_path in (root / 'sensitivity').glob('*/splits.csv'):
        df = pd.read_csv(split_path)
        if split_path.parent.name.startswith('grouped_'):
            partitions = df.loc[df.partition != 'unused_train'].groupby('group').partition.nunique()
            assert partitions.max() == 1
    xml = ET.parse(root / 'tests.xml').getroot()
    cases = xml.findall('.//testcase')
    assert not xml.findall('.//failure') and not xml.findall('.//error')
    skipped = len(xml.findall('.//skipped'))
    result = {'status': 'passed', 'stages': checks, 'tests_passed': len(cases)-skipped,
              'tests_skipped': skipped, 'supplement_model_fits': len(sensitivity),
              'bootstrap_prefix_preserved': True, 'grouped_split_overlap': 0}
    (root / 'VERIFIED.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    quality = pd.read_csv(root / 'phase1_prediction/quality_summary.csv')
    quality = quality.query("scenario in ['n_2000', 'base', 'n_12000'] and engine != 'prevalence'")
    quality = quality[['scenario', 'engine', 'roc_auc_mean', 'pr_auc_ap_mean', 'brier_mean', 'f1_test_at_valid_threshold_mean']]
    timing = pd.read_csv(root / 'phase2_parallel/timing_summary.csv')
    timing = timing[['engine', 'mode', 'seconds_median', 'process_seconds_median', 'speedup', 'process_speedup', 'rss_mib_max']]
    supplement = sensitivity.groupby(['split', 'model'])[['roc_auc','pr_auc_ap','brier','f1_test_at_valid_threshold']].mean().reset_index()
    report = '# Windows 实跑结果\n\n'
    report += '本页由 scripts/verify_windows.py 从原始结果复算生成。验证详情见 VERIFIED.json；原 macOS 结果单独保留。\n\n'
    report += '## 完整性\n\n' + markdown_table(pd.DataFrame(checks)) + '\n\n'
    report += f'测试：{len(cases)-skipped} 通过、{skipped} 跳过（POSIX 用例）。B=40 的前 20 份抽样和模型种子与 B=20 完全相同。分组划分的训练、验证、测试之间没有重复特征组交叉。\n\n'
    report += '## 阶段一：自然比例预测指标\n\n' + markdown_table(quality) + '\n\n'
    report += '## 阶段二：计时与内存\n\n' + markdown_table(timing) + '\n\n'
    report += '时间单位为秒；内存为 MiB。speedup 使用核心时间，process_speedup 使用完整进程时间。两者都与同一引擎的串行中位数比较。\n\n'
    report += '## 补充对照：三个划分种子的均值\n\n' + markdown_table(supplement) + '\n\n'
    report += '该对照不使用外层 Bootstrap。分组划分也改变了测试集，不能将两种划分的指标差异全部归因于重复记录；3 个种子只提供描述性敏感性检查。详细标准差、逐次预测与划分在 sensitivity/。\n\n'
    report += '## 校准与规模变化\n\n阶段一可靠性曲线：\n\n![Calibration](phase1_prediction/figures/calibration.png)\n\n'
    report += '曲线预先选择 bootstrap_id=0；分箱样本量见 phase1_prediction/calibration.csv。Brier 为概率均方误差，不是纯校准指标。规模变化见 phase3_scaling/timing_summary.csv；分裂内核见 phase2_parallel/kernel/summary.csv。\n'
    (root / 'WINDOWS_RESULTS.md').write_text(report, encoding='utf-8')
    entries = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'ARTIFACT_SHA256.json'}
    (root / 'ARTIFACT_SHA256.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print('Artifact files:', len(entries))


if __name__ == '__main__':
    verify(sys.argv[1])
