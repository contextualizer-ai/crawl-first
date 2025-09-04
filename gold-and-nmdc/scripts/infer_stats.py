#!/usr/bin/env python3
import argparse, json, math, collections
from typing import Any, Dict, List, Tuple
from pymongo import MongoClient
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Compass-like field stats from MongoDB collection")
    ap.add_argument("--mongo-uri", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--coll", required=True)
    ap.add_argument("--sample-size", type=int, default=50000)
    ap.add_argument("--query", default="{}",
                    help='JSON string filter, e.g. \'{"mixsPackage":"Soil"}\'')
    ap.add_argument("--max-examples", type=int, default=3,
                    help="Max example values to keep per field")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    return ap.parse_args()

def typeof(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int) and not isinstance(v, bool):
        return "integer"
    if isinstance(v, float):
        # Mongo's special doubles (nan, inf) may appear
        if math.isnan(v): return "number(NaN)"
        if v == float("inf"): return "number(Inf)"
        if v == float("-inf"): return "number(-Inf)"
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__

def walk(doc: Any, prefix: str, seen: Dict[str, dict], max_examples: int, doc_id: str):
    """Recursively record stats for each path."""
    if isinstance(doc, dict):
        for k, v in doc.items():
            path = f"{prefix}.{k}" if prefix else k
            stats = seen.setdefault(path, {
                "docs_with_field": set(),  # Track which docs have this field NON-NULL
                "types": collections.Counter(),
                "examples": collections.OrderedDict(),  # preserves insertion
                "array_elem_types": collections.Counter(),  # only for arrays
            })
            # Only count as "present" if value is not null
            if v is not None:
                stats["docs_with_field"].add(doc_id)
            t = typeof(v)
            stats["types"][t] += 1
            if len(stats["examples"]) < max_examples:
                try:
                    key = json.dumps(v, sort_keys=True, default=str)
                except Exception:
                    key = str(v)
                stats["examples"][key] = None
            # Recurse
            if isinstance(v, dict):
                walk(v, path, seen, max_examples, doc_id)
            elif isinstance(v, list):
                for elem in v:
                    et = typeof(elem)
                    stats["array_elem_types"][et] += 1
                    if isinstance(elem, (dict, list)):
                        walk(elem, path + "[]", seen, max_examples, doc_id)

def main():
    args = parse_args()
    query = json.loads(args.query)

    client = MongoClient(args.mongo_uri)
    coll = client[args.db][args.coll]

    pipeline = []
    if query:
        pipeline.append({"$match": query})
    pipeline.append({"$sample": {"size": args.sample_size}})

    seen: Dict[str, dict] = {}
    n_docs = 0
    for doc in coll.aggregate(pipeline, allowDiskUse=True):
        n_docs += 1
        doc_id = str(doc.get("_id", f"doc_{n_docs}"))
        walk(doc, "", seen, args.max_examples, doc_id)

    rows: List[Dict[str, Any]] = []
    for path, s in seen.items():
        # Coverage = percentage of documents that have this field (non-null)
        docs_with_field = len(s["docs_with_field"])
        coverage = (docs_with_field / n_docs * 100.0) if n_docs else 0.0
        types_summary = ", ".join(f"{k}:{v}" for k, v in s["types"].most_common())
        arr_types = ", ".join(f"{k}:{v}" for k, v in s["array_elem_types"].most_common()) if s["array_elem_types"] else ""
        examples = []
        for ex in list(s["examples"].keys()):
            if len(ex) > 120:
                ex = ex[:117] + "…"
            examples.append(ex)
        rows.append({
            "field_path": path,
            "coverage_pct": round(coverage, 3),
            "present_docs": docs_with_field,
            "types_seen": types_summary,
            "array_elem_types": arr_types,
            "examples": " | ".join(examples),
        })

    df = pd.DataFrame(rows).sort_values(["field_path"]).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)
    with open(args.out_md, "w") as f:
        f.write(f"# Field statistics for {args.db}.{args.coll}\n\n")
        f.write(f"- Sampled docs: **{n_docs}**\n")
        f.write(f"- Query: `{json.dumps(query)}`\n\n")
        f.write(df.to_markdown(index=False))
    print(f"Wrote stats CSV → {args.out_csv}")
    print(f"Wrote stats MD  → {args.out_md}")

if __name__ == "__main__":
    main()