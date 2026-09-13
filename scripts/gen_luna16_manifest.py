"""Generate the LUNA16 fixed patient-level 7:1:2 split manifest.

LUNA16 = a curated subset of LIDC-IDRI.  A single LIDC-IDRI patient can
contribute more than one CT series (e.g. LIDC-IDRI-0332 has two series in
LUNA16).  To honour the manuscript's "fixed patient-level split" (Section 4.1)
we partition at the *PatientID* level and only expand to SeriesInstanceUIDs
(the on-disk .mhd stems) when writing the manifest, so that no patient leaks
across folds.

The output ``{train,val,test}_patients.txt`` lists SeriesInstanceUIDs (what
``scripts/preprocess.py`` and ``scripts/train.py`` key slices on), while
``seriesuid_to_patient.csv`` documents the patient grouping for transparency.

The partition is fully reproducible: it uses the same deterministic algorithm
as ``scripts/preprocess.patient_split`` with the INDEPENDENT split seed
(``common.split_seed`` = 42), decoupled from the optimisation seeds 42-46.

Usage
-----
    python scripts/gen_luna16_manifest.py \
        --mapping /path/to/lidc_ct_series.json \
        --seriesuids /path/to/luna16_official_888_seriesuids.txt

If ``--mapping`` is omitted, the committed ``seriesuid_to_patient.csv`` is used
(the seriesuid list is then taken from that CSV).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPLIT_DIR = REPO / "data" / "splits" / "luna16"
SPLIT_SEED = 42
RATIOS = (0.7, 0.1, 0.2)


def patient_split(patient_ids, ratios=RATIOS, seed=SPLIT_SEED):
    """Identical algorithm to scripts/preprocess.patient_split."""
    ids = sorted(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }


def load_mapping_from_json(path: Path, seriesuids):
    data = json.loads(path.read_text(encoding="utf-8"))
    series = data if isinstance(data, list) else data.get("series", data)
    s2p = {}
    for s in series:
        sid = s.get("SeriesInstanceUID") or s.get("SeriesId")
        pid = s.get("PatientID") or s.get("PatientId")
        if sid and pid:
            s2p[sid] = pid
    return {s: s2p[s] for s in seriesuids}


def load_mapping_from_csv(path: Path):
    m = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            m[row["seriesuid"].strip()] = row["patient_id"].strip()
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping", type=Path, default=None,
                    help="TCIA NBIA series->patient JSON (only needed for the "
                         "first generation; afterwards the committed CSV suffices)")
    ap.add_argument("--seriesuids", type=Path, default=None,
                    help="newline list of the 888 official SeriesInstanceUIDs")
    args = ap.parse_args()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mapping is not None and args.seriesuids is not None:
        seriesuids = [l.strip() for l in args.seriesuids.read_text().splitlines()
                      if l.strip()]
        s2p = load_mapping_from_json(args.mapping, seriesuids)
    else:
        csv_path = SPLIT_DIR / "seriesuid_to_patient.csv"
        if not csv_path.exists():
            raise SystemExit("Neither --mapping nor a committed "
                             "seriesuid_to_patient.csv was found.")
        s2p = load_mapping_from_csv(csv_path)
        seriesuids = sorted(s2p)

    missing = [s for s in seriesuids if s not in s2p]
    if missing:
        raise SystemExit(f"{len(missing)} seriesuids could not be mapped to a "
                         f"PatientID (e.g. {missing[:3]})")

    # group seriesuids by PatientID
    pat_to_series: dict[str, list[str]] = {}
    for s in seriesuids:
        pat_to_series.setdefault(s2p[s], []).append(s)
    patients = sorted(pat_to_series)
    print(f"[gen] seriesuids={len(seriesuids)}  patients={len(patients)}")

    splits = patient_split(patients, seed=SPLIT_SEED)

    # write the patient->seriesuid mapping (public dataset IDs, no PII)
    with (SPLIT_DIR / "seriesuid_to_patient.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seriesuid", "patient_id"])
        for s in seriesuids:
            w.writerow([s, s2p[s]])

    # write the per-fold manifest files (SeriesInstanceUIDs, sorted)
    stats = {}
    all_suids: set[str] = set()
    for name, pats in splits.items():
        suids = []
        for p in pats:
            suids.extend(sorted(pat_to_series[p]))
        all_suids |= set(suids)
        (SPLIT_DIR / f"{name}_patients.txt").write_text(
            "\n".join(sorted(suids)) + "\n")
        stats[name] = (len(pats), len(suids))

    # ---- validation ----
    pat_sets = {k: set(v) for k, v in splits.items()}
    for a in pat_sets:
        for b in pat_sets:
            if a < b:
                leak = pat_sets[a] & pat_sets[b]
                assert not leak, f"patient leakage between {a} and {b}: {leak}"
    assert all_suids == set(seriesuids), "manifest union != official seriesuid set"
    # the dual-series patient must be fully inside one fold
    dual = [p for p, ss in pat_to_series.items() if len(ss) > 1]
    for p in dual:
        fold_of = {name: p in set(pats) for name, pats in splits.items()}
        assert sum(fold_of.values()) == 1, f"dual-series patient {p} split across folds"

    print("[gen] patient-level counts (patients): "
          f"train={stats['train'][0]} val={stats['val'][0]} test={stats['test'][0]}")
    print("[gen] seriesuid-level counts           : "
          f"train={stats['train'][1]} val={stats['val'][1]} test={stats['test'][1]}")
    print(f"[gen] dual-series patients kept together: {dual}")
    print("[gen] VALIDATION OK: no patient leakage, union complete, "
          "dual-series patients intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
