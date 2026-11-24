"""Import the pinned Mozilla and EDC sources using their published mappings."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import re

from .common import ROOT, file_hash, write_json

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SOURCES = {
    "mozilla": {
        "citation": "Burak Yolacan (2024). Mozilla browser front-end software requirements and test scripts dataset, V1.",
        "source": "https://doi.org/10.17632/jw5bgshtvg.1",
        "license": "CC BY 4.0",
        "files": {
            "R-set.csv": "2910ee44ec7d09bd75ba89dc99d0ec7f12fa128a93109c2f326ec643f4515688",
            "T-set.csv": "bfa2dcb26c3329f60c8cbc97d2ae67d60a172b7fef282eac40b1429f100aac4e",
        },
    },
    "edc": {
        "citation": "Jorge Gavilan (2025). Requirements Traceability Matrix - EDC, v1.",
        "source": "https://doi.org/10.5281/zenodo.17546974",
        "license": "CC BY 4.0",
        "files": {
            "edc.xlsx": "2d0275370b8553c3c525f19d852655e4da573b58999c8082f73016f7683f840c",
        },
    },
}
REQUIREMENT_ID = re.compile(r"DSP-(?:CAT|CN|TP)-\d+")
TEST_ID = re.compile(r"DSP-TC-\d+")
ANY_ID = re.compile(r"\b(?:DSP-(?:CAT|CN|TP|TC)|R|TC)-\d+\b", re.I)


def clean(parts):
    text = " ".join(str(value) for value in parts if value is not None)
    return " ".join(ANY_ID.sub(" ", text).split())


def add_text(records, key, text):
    if key in records or not text:
        raise ValueError(f"duplicate ID or empty text: {key}")
    records[key] = text


def mozilla(source_dir):
    artifacts = []
    for name, header, id_pattern in (
        ("R-set.csv", ["RequirementID", "RequirementContent"], r"R-\d+"),
        ("T-set.csv", ["TestCaseID", "TestCaseContent"], r"TC-\d+"),
    ):
        records = {}
        with (source_dir / name).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter=";", strict=True)
            if next(reader) != header:
                raise ValueError(f"unexpected header in {name}")
            for key, text in reader:
                key = key.strip()
                if not re.fullmatch(id_pattern, key):
                    raise ValueError(f"unexpected ID in {name}: {key}")
                add_text(records, key, clean([text]))
        artifacts.append(records)
    requirements, cases = artifacts
    links = {(requirement, "TC-" + requirement[2:]) for requirement in requirements}
    if {test_id for _, test_id in links} != cases.keys():
        raise ValueError("Mozilla case IDs do not match the documented requirement IDs")
    return requirements, cases, links, {"mapping": "published one-to-one R-XYZ to TC-XYZ"}


def edc(source_dir):
    from openpyxl import load_workbook

    requirements, cases = {}, {}
    from_requirements, from_cases = set(), set()
    workbook = load_workbook(source_dir / "edc.xlsx", read_only=True, data_only=True)
    try:
        for sheet in workbook:
            rows = sheet.iter_rows(max_col=7, values_only=True)
            header = next(rows)
            if header[0] == "Requirement ID" and header[5] == "TC IDs":
                for row in rows:
                    key = str(row[0] or "").strip()
                    if REQUIREMENT_ID.fullmatch(key):
                        add_text(requirements, key, clean(row[1:4]))
                        from_requirements.update((key, id_) for id_ in TEST_ID.findall(str(row[5])))
            elif header[:3] == ("Test Case ID", "Requirement ID", "Description"):
                for row in rows:
                    key = str(row[0] or "").strip()
                    if TEST_ID.fullmatch(key):
                        add_text(cases, key, clean(row[2:3]))
                        from_cases.update((id_, key) for id_ in REQUIREMENT_ID.findall(str(row[1])))
            else:
                raise ValueError(f"unexpected EDC sheet layout: {sheet.title}")
    finally:
        workbook.close()
    union = from_requirements | from_cases
    if any(r not in requirements or t not in cases for r, t in union):
        raise ValueError("EDC contains unresolved links")
    disagreements = from_requirements ^ from_cases
    excluded = {r for r, _ in disagreements}
    links = {(r, t) for r, t in from_requirements & from_cases if r not in excluded}
    report = {
        "requirement_sheet_only": sorted(from_requirements - from_cases),
        "test_sheet_only": sorted(from_cases - from_requirements),
        "excluded_disagreement_requirements": sorted(excluded),
    }
    return requirements, cases, links, report


def write_table(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(sorted(rows))


def import_data(name, source_dir=None):
    source = SOURCES[name]
    source_dir = source_dir or RAW / name
    for filename, expected in source["files"].items():
        if file_hash(source_dir / filename) != expected:
            raise ValueError(f"SHA-256 mismatch for {filename}")
    loader = {"mozilla": mozilla, "edc": edc}[name]
    requirements, cases, links, mapping = loader(source_dir)
    output = PROCESSED / name
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "requirements.csv", ("id", "text"), requirements.items())
    write_table(output / "test_cases.csv", ("id", "text"), cases.items())
    write_table(output / "reference_links.csv", ("requirement_id", "test_case_id"), links)
    counts = Counter(cases.values())
    report = {
        **source, **mapping,
        "requirements": len(requirements),
        "test_cases": len(cases),
        "reference_links": len(links),
        "excluded_query_ids": sorted(requirements.keys() - {r for r, _ in links}),
        "duplicate_case_text_groups": sum(value > 1 for value in counts.values()),
    }
    write_json(output / "import_report.json", report)
    print(f"{name}: {len(requirements)} requirements, {len(cases)} cases, {len(links)} links")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    import_data(args.dataset, args.source_dir)
