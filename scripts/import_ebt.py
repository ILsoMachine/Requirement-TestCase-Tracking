"""Validate the pinned TRIAD EBT files and produce retrieval and scoring data."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY = "https://github.com/huiAlex/TRIAD"
SOURCE_COMMIT = "524929e892dd39b8b370ffeef17b4c6962210e07"
SOURCE_FILES = {
    "Requirements.csv": (
        "dataset/ebt/originals/requirements/Requirements.csv",
        "e38462cccce24baa02797a47e15fc20bc876735198a27dc867ad841988379781",
    ),
    "TestCases.csv": (
        "dataset/ebt/originals/testcases/TestCases.csv",
        "be428c235ac8f3b3a59cfe420c1c0ecedad84fa4cdb96775902c01002a323689",
    ),
    "req-tc.txt": (
        "dataset/ebt/trace_matrices/req-tc.txt",
        "b48d2d69d75d228c9910a23a9d92c7f31ed1e6dcd37a80519f87c922788b2002",
    ),
}
TITLE_REFERENCES = re.compile(
    r"(?:\s*\([0-9]+(?:\.[0-9]+)+\.?\??\))+(?=\s+pre\s*conditions\b)",
    re.IGNORECASE,
)


def read_sources(raw_dir):
    texts = {}
    provenance = {}
    for name, (source_path, expected_hash) in SOURCE_FILES.items():
        path = raw_dir / name
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        if digest != expected_hash:
            raise ValueError(f"{path}: SHA-256 mismatch ({digest})")
        texts[name] = raw.decode("utf-8")
        provenance[name] = {
            "source_path": source_path,
            "sha256": digest,
            "bytes": len(raw),
            "encoding": "UTF-8 (ASCII subset)",
            "line_endings": "LF",
            "format": "comma-separated CSV" if name.endswith(".csv") else "whitespace-separated",
        }
    return texts, provenance


def parse_artifacts(name, source, *, remove_title_references=False):
    reader = csv.reader(StringIO(source, newline=""), strict=True)
    if next(reader, None) != ["id", "text"]:
        raise ValueError(f"{name}: expected the header id,text")

    artifacts = {}
    changed_count = 0
    for row in reader:
        location = f"{name}:{reader.line_num}"
        if len(row) != 2:
            raise ValueError(f"{location}: expected an ID and text")
        artifact_id, text = row
        if not re.fullmatch(r"[0-9]+", artifact_id):
            raise ValueError(f"{location}: invalid ID {artifact_id!r}")
        if artifact_id in artifacts:
            raise ValueError(f"{location}: duplicate ID {artifact_id}")
        if remove_title_references:
            text, removed = TITLE_REFERENCES.subn("", text)
            changed_count += bool(removed)
        clean_text = " ".join(text.split())
        if not clean_text:
            raise ValueError(f"{location}: empty text for {artifact_id}")
        artifacts[artifact_id] = clean_text

    return artifacts, changed_count


def parse_links(source, requirements, test_cases):
    links = set()
    for line_no, line in enumerate(source.splitlines(), start=1):
        parts = line.split()
        location = f"req-tc.txt:{line_no}"
        if len(parts) != 2:
            raise ValueError(f"{location}: expected a requirement ID and a test case ID")
        requirement_id, test_case_id = parts
        if requirement_id not in requirements:
            raise ValueError(f"{location}: unknown requirement ID {requirement_id}")
        if test_case_id not in test_cases:
            raise ValueError(f"{location}: unknown test case ID {test_case_id}")
        link = (requirement_id, test_case_id)
        if link in links:
            raise ValueError(f"{location}: duplicate link {requirement_id} {test_case_id}")
        links.add(link)
    if not links:
        raise ValueError("req-tc.txt: no reference links")
    return sorted(links)


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def import_ebt(raw_dir, output_dir):
    sources, provenance = read_sources(raw_dir)
    requirements, _ = parse_artifacts("Requirements.csv", sources["Requirements.csv"])
    test_cases, changed_count = parse_artifacts(
        "TestCases.csv", sources["TestCases.csv"], remove_title_references=True
    )
    links = parse_links(sources["req-tc.txt"], requirements, test_cases)
    links_per_requirement = Counter(requirement_id for requirement_id, _ in links)
    linked_test_cases = {test_case_id for _, test_case_id in links}
    unlinked_requirements = sorted(requirements.keys() - links_per_requirement.keys())
    report = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_files": provenance,
        "counts": {
            "requirements": len(requirements),
            "test_cases": len(test_cases),
            "reference_links": len(links),
            "requirements_with_reference_links": len(links_per_requirement),
            "requirements_without_reference_links": len(unlinked_requirements),
            "test_cases_with_reference_links": len(linked_test_cases),
            "test_cases_without_reference_links": len(test_cases) - len(linked_test_cases),
            "test_cases_with_title_references_removed": changed_count,
        },
        "requirement_ids_without_reference_links": unlinked_requirements,
        "reference_link_counts_over_five": {
            requirement_id: count
            for requirement_id, count in sorted(links_per_requirement.items())
            if count > 5
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "requirements.csv", ("id", "text"), sorted(requirements.items()))
    write_csv(output_dir / "test_cases.csv", ("id", "text"), sorted(test_cases.items()))
    write_csv(
        output_dir / "reference_links.csv", ("requirement_id", "test_case_id"), links
    )
    (output_dir / "import_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "ebt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "ebt")
    args = parser.parse_args()

    if args.raw_dir.resolve() == args.output_dir.resolve():
        parser.error("the raw and output directories must be different")
    try:
        report = import_ebt(args.raw_dir, args.output_dir)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"EBT import failed: {error}\n")

    for name, count in report["counts"].items():
        print(f"{name}: {count}")
    print(args.output_dir / "import_report.json")


if __name__ == "__main__":
    main()
