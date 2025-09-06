#!/usr/bin/env python3
"""
Crosswalk compiler: CSV → SSSOM with ENVO CURIE resolution.

Loads human-editable CSV mappings and compiles them to SSSOM TSV format,
automatically resolving ENVO CURIEs via label matching when left blank.
"""

import csv
import sys
import datetime
import json
import pathlib
from collections import defaultdict

ENVO_TSV = pathlib.Path("../data/envo/envo_labels.tsv")
OUT_DIR = pathlib.Path("../mappings_sssom")
OUT_DIR.mkdir(exist_ok=True)

def load_envo_index(tsv=ENVO_TSV):
    """Load ENVO terms into searchable index by normalized labels."""
    idx = defaultdict(set)  # normalized label -> {(id,label)}
    
    if not tsv.exists():
        print(f"Warning: ENVO index not found at {tsv}")
        print("Run 'make envo/extract' to build the ENVO label index")
        return idx
        
    with open(tsv, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            term_id = row.get("ID") or row.get("id")
            label = row.get("LABEL") or row.get("label")
            syns = (row.get("SYNONYMS") or "") + "|" + (row.get("ALT") or "")
            
            # Index by primary label and all synonyms
            for s in filter(None, {label, *[x.strip() for x in syns.split("|")]}):
                key = " ".join(s.lower().split())
                idx[key].add((term_id, label))
    
    print(f"Loaded {len(idx)} ENVO terms from {tsv}")
    return idx

def resolve_curie(envo_idx, envo_label, envo_curie):
    """Resolve ENVO CURIE from label if not provided."""
    if envo_curie:  # trust provided CURIE
        return envo_curie, envo_label
    if not envo_label:
        return "", ""
    
    # Normalize label for lookup
    key = " ".join(envo_label.lower().split())
    candidates = sorted(envo_idx.get(key, []))
    
    if candidates:
        return candidates[0][0], candidates[0][1]  # first match (deterministic)
    else:
        print(f"Warning: No ENVO match found for '{envo_label}'")
        return "", envo_label

def compile_csv(csv_path, envo_idx):
    """Compile CSV mapping to SSSOM rows."""
    sssom_rows = []
    
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            curie, label = resolve_curie(envo_idx, row["envo_label"], row.get("envo_curie", ""))
            pred = row.get("relation", "skos:closeMatch")
            
            sssom_rows.append({
                "subject_id": f'{row["source_system"]}:{row["source_code"]}',
                "subject_label": row["source_label"],
                "object_id": curie,
                "object_label": label,
                "predicate_id": pred,
                "mapping_justification": "semapv:UnspecifiedMatching",
                "mapping_tool": "crosswalk-compiler/0.1",
                "mapping_date": datetime.date.today().isoformat(),
                "creator_id": row.get("editor_id", ""),
                "see_also": row.get("source_url", ""),
                "comment": row.get("notes", ""),
            })
    
    return sssom_rows

def write_sssom(sssom_rows, out_path):
    """Write SSSOM TSV file."""
    fields = [
        "subject_id", "subject_label", "predicate_id", "object_id", "object_label",
        "mapping_justification", "mapping_tool", "mapping_date", "creator_id", 
        "see_also", "comment"
    ]
    
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(sssom_rows)
    
    print(f"Written {len(sssom_rows)} mappings to {out_path}")

def main():
    """Compile all CSV mappings to SSSOM."""
    envo_idx = load_envo_index()
    
    # Process all mapping CSV files
    mapping_files = ["usda_texture_12.csv", "osm_natural.csv", "wrb_groups.csv"]
    
    for csv_name in mapping_files:
        csv_path = pathlib.Path("mappings") / csv_name
        if not csv_path.exists():
            print(f"Skipping {csv_name} (not found)")
            continue
            
        print(f"Compiling {csv_name}...")
        rows = compile_csv(csv_path, envo_idx)
        out_path = OUT_DIR / csv_name.replace(".csv", ".sssom.tsv")
        write_sssom(rows, out_path)

if __name__ == "__main__":
    main()