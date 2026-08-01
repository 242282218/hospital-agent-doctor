#!/usr/bin/env python
"""Expand verified case-memory toward 300 via repeated train GT batches.

Usage (from hospital_agent_example, with local.env loaded):
  .venv\\Scripts\\python.exe scripts/memory/batch_expand_train_gt_to_300.py \\
      --target 300 --batch-size 5 --max-batches 50 --start-seed 2026072410

Does NOT switch production pointer. Incomplete cases are not retried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAT = re.compile(r"Patient_(?:Comorbid-)?[A-Za-z0-9_-]+")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_latest_release() -> Path | None:
    releases = sorted(
        ROOT.glob("releases/release_C_case_memory_*cases"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for rel in releases:
        reg = rel / "verified_registry.json"
        if reg.exists():
            return rel
    return None


def frozen_patient_ids(release_dir: Path) -> set[str]:
    reg = load_json(release_dir / "verified_registry.json")
    return {a["content"]["patient_id"] for a in reg.get("assets", [])}


def history_patient_ids() -> set[str]:
    ids: set[str] = set()
    for mode in ("test", "train"):
        root = ROOT / "outputs" / mode
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            try:
                ids.update(PAT.findall(path.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                pass
    return ids


def fetch_patients(seed: int, count: int = 100) -> list[str]:
    base = os.environ["SERVICE_BASE_URL"].rstrip("/")
    token = os.environ["SERVICE_TRAIN_TOKEN"]
    team = os.environ["TEAM_ID"]
    url = (
        f"{base}/patients?team_id={team}&patient_count={count}"
        f"&selection=random&random_seed={seed}"
    )
    req = urllib.request.Request(url, headers={"X-Hospital-Service-Token": token})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    ids = data.get("patient_ids") if isinstance(data, dict) else data
    return list(ids)


def write_config(patient_ids: list[str], seed: int) -> None:
    lines = [
        "# Baseline agent configuration.",
        "output_dir: outputs",
        "train:",
        "  selection: random",
        f"  patient_count: {len(patient_ids)}",
        f"  random_seed: {seed}",
        "  patient_ids:",
    ]
    for pid in patient_ids:
        lines.append(f"    - {pid}")
    lines += [
        "test:",
        "  selection: random",
        "  patient_count: 1",
        "  random_seed: 42",
        "  patient_ids: []",
        "memory:",
        "  md_path: data/memory_data/memory.md",
        "  max_notes: 3",
        "  max_note_chars: 1200",
        "log_llm_prompts: false",
        "",
    ]
    (ROOT / "config.yaml").write_text("\n".join(lines), encoding="utf-8")


def restore_config() -> None:
    write_config([], 42)
    # empty patient_ids list format
    text = """# Baseline agent configuration.
output_dir: outputs
train:
  selection: random
  patient_count: 1
  random_seed: 42
  patient_ids: []
test:
  selection: random
  patient_count: 1
  random_seed: 42
  patient_ids: []
memory:
  md_path: data/memory_data/memory.md
  max_notes: 3
  max_note_chars: 1200
log_llm_prompts: false
"""
    (ROOT / "config.yaml").write_text(text, encoding="utf-8")


def run_train() -> int:
    return subprocess.call([sys.executable, "train.py"], cwd=str(ROOT))


def code_commit_label() -> str:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    # light dirty marker from agent sources
    h = hashlib.sha256()
    for pattern in ("agent/**/*.py", "offline/**/*.py", "data/ref_data/*.json", "config.yaml"):
        for p in ROOT.glob(pattern):
            if p.is_file() and "__pycache__" not in str(p):
                h.update(p.read_bytes())
    return f"{head}+dirty:{h.hexdigest()[:16]}"


def import_and_build(release_tag: str, batch_size: int, seed: int) -> tuple[int, Path | None]:
    from offline.train_trust import build_train_trust_manifest, write_train_trust_manifest

    base_path = ROOT / "docs" / "架构迁移基线" / "manifest.json"
    base = load_json(base_path) if base_path.exists() else None
    m = build_train_trust_manifest(train_outputs=ROOT / "outputs" / "train", base_manifest=base)
    trust_dir = ROOT / "outputs" / "offline" / "phase0_20260723"
    trust_dir.mkdir(parents=True, exist_ok=True)
    trust_path = trust_dir / f"train_trust_{m['manifest_hash'][:16]}.json"
    if not trust_path.exists():
        write_train_trust_manifest(m, path=trust_path)

    import_cmd = [
        sys.executable,
        "-m",
        "scripts.memory.import_case_memory_candidates",
        "--train-outputs",
        "outputs/train",
        "--trust-manifest",
        str(trust_path.relative_to(ROOT)),
        "--artifact-root",
        "outputs/offline/case_memory/imports",
    ]
    proc = subprocess.run(import_cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print("IMPORT_FAIL", proc.stderr[-500:] if proc.stderr else proc.stdout[-500:])
        return -1, None
    # parse last JSON line
    out = (proc.stdout or "").strip().splitlines()
    payload = None
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if not payload:
        print("IMPORT_NO_JSON")
        return -1, None
    selected = int(payload.get("manifest", {}).get("selected_count") or 0)
    batch_dir = Path(payload["batch_dir"])
    if not batch_dir.is_absolute():
        batch_dir = ROOT / batch_dir
    print(f"import selected={selected} conflicts={payload.get('manifest', {}).get('ground_truth_conflict_count')}")

    release_dir = ROOT / "releases" / release_tag
    build_cmd = [
        sys.executable,
        "-m",
        "scripts.memory.build_case_memory_release",
        "--import-batch",
        str(batch_dir.relative_to(ROOT)),
        "--base-release",
        "releases/release_A_quality",
        "--release-dir",
        str(release_dir.relative_to(ROOT)),
        "--reviewer",
        "github:24228",
        "--ref-data-dir",
        "data/ref_data",
        "--knowledge-dir",
        "agent/knowledge",
        "--code-commit",
        code_commit_label(),
        "--production-pointer",
        "releases/current.json",
    ]
    proc2 = subprocess.run(build_cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(proc2.stdout[-400:] if proc2.stdout else "")
    if proc2.returncode != 0:
        print("BUILD_FAIL", proc2.stderr[-500:] if proc2.stderr else "")
        return selected, None
    # verify pointer unchanged
    ptr = sha256_file(ROOT / "releases" / "current.json")
    print("pointer_sha", ptr[:16], "unchanged_expected")
    return selected, release_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--max-batches", type=int, default=60)
    ap.add_argument("--start-seed", type=int, default=2026072410)
    ap.add_argument("--commit", action="store_true", help="git commit each release")
    args = ap.parse_args()

    for key in ("SERVICE_BASE_URL", "SERVICE_TRAIN_TOKEN", "TEAM_ID"):
        if not os.environ.get(key):
            print(f"missing env {key}; source local.env.ps1 first")
            return 2

    log_path = ROOT / "outputs" / "offline" / "phase0_20260723" / "expand_to_300.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    for i in range(args.max_batches):
        rel = find_latest_release()
        if not rel:
            log("NO_RELEASE")
            return 1
        frozen = frozen_patient_ids(rel)
        n = len(frozen)
        log(f"batch={i} frozen={n}/{args.target} release={rel.name}")
        if n >= args.target:
            log(f"TARGET_REACHED {n}")
            restore_config()
            return 0

        exclude = frozen | history_patient_ids()
        seed = args.start_seed + i
        pool = fetch_patients(seed, count=100)
        pick = [p for p in pool if p not in exclude][: args.batch_size]
        if len(pick) < args.batch_size:
            # try another seed
            pool2 = fetch_patients(seed + 1000, count=200)
            pick = [p for p in pool2 if p not in exclude][: args.batch_size]
        if len(pick) < 1:
            log("NO_NEW_PATIENTS")
            return 1
        log(f"pick={pick}")
        write_json(
            ROOT / "outputs" / "selections" / f"train_gt_auto_{seed}.json",
            {"seed": seed, "patient_ids": pick, "batch": i},
        )
        write_config(pick, seed)
        rc = run_train()
        restore_config()
        if rc != 0:
            log(f"TRAIN_FAIL rc={rc}")
            # continue rather than abort whole loop
            continue

        tag = f"release_C_case_memory_20260724_v_auto_{n + len(pick)}cases"
        # after import selected may be less than n+len(pick) if incomplete
        selected, release_dir = import_and_build(tag, args.batch_size, seed)
        if release_dir is None:
            log("BUILD_SKIP")
            continue
        assets = len(load_json(release_dir / "verified_registry.json").get("assets", []))
        log(f"frozen_now={assets}")
        if args.commit:
            subprocess.call(["git", "add", str(release_dir.relative_to(ROOT))], cwd=str(ROOT))
            msg = f"feat(memory): auto freeze case-memory {assets}/300"
            subprocess.call(["git", "commit", "-m", msg], cwd=str(ROOT))

        if assets >= args.target:
            log(f"TARGET_REACHED {assets}")
            restore_config()
            return 0

    restore_config()
    rel = find_latest_release()
    n = len(frozen_patient_ids(rel)) if rel else 0
    log(f"MAX_BATCHES_DONE frozen={n}")
    return 0 if n >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
