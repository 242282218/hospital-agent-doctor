from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from agent.legacy_orchestrator import (
    flatten_disease_catalog,
    flatten_examination_catalog,
    load_disease_catalog,
    load_examination_catalog,
)
from offline.artifacts import file_hash
from offline.case_memory_import import import_case_memory_candidates


BASE_DIR = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import trusted train evaluations as case-memory candidates.")
    parser.add_argument("--train-outputs", type=Path, required=True)
    parser.add_argument("--trust-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--ref-data-dir", type=Path, default=BASE_DIR / "data" / "ref_data")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ref_data_dir = Path(args.ref_data_dir)
    disease_path = ref_data_dir / "diseases_catalog.json"
    examination_path = ref_data_dir / "examinations_catalog.json"
    if ref_data_dir.resolve() == (BASE_DIR / "data" / "ref_data").resolve():
        diseases = flatten_disease_catalog(load_disease_catalog())
        examinations = flatten_examination_catalog(load_examination_catalog())
    else:
        disease_data = json.loads(disease_path.read_text(encoding="utf-8"))
        diseases = [
            str(name)
            for names in disease_data.get("diseases", {}).values()
            for name in names
            if str(name).strip()
        ]
        examination_data = json.loads(examination_path.read_text(encoding="utf-8"))
        examinations = [
            str(item.get("name") if isinstance(item, dict) else item)
            for items in examination_data.get("examinations", {}).values()
            for item in items
            if str(item.get("name") if isinstance(item, dict) else item).strip()
        ]
    return import_case_memory_candidates(
        train_outputs=args.train_outputs,
        trust_manifest_path=args.trust_manifest,
        artifact_root=args.artifact_root,
        official_diseases=diseases,
        valid_examinations=examinations,
        catalog_hashes={
            "diseases_catalog.json": file_hash(disease_path),
            "examinations_catalog.json": file_hash(examination_path),
        },
    )


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
