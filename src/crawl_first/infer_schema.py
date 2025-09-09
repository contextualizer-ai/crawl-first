#!/usr/bin/env python3
import argparse
import json
import sys

from genson import SchemaBuilder
from pymongo import MongoClient


def parse_args():
    ap = argparse.ArgumentParser(
        description="Infer JSON Schema from MongoDB collection"
    )
    ap.add_argument("--mongo-uri", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--coll", required=True)
    ap.add_argument("--sample-size", type=int, default=50000)
    ap.add_argument(
        "--query",
        default="{}",
        help='JSON string filter, e.g. \'{"mixsPackage":"Soil"}\'',
    )
    ap.add_argument("--out-json-schema", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        query = json.loads(args.query)
    except Exception as e:
        print(f"Bad --query JSON: {e}", file=sys.stderr)
        sys.exit(2)

    client = MongoClient(args.mongo_uri)
    coll = client[args.db][args.coll]

    # Server-side sampling to reduce bias/IO; apply query first if provided.
    pipeline = []
    if query:
        pipeline.append({"$match": query})
    pipeline.append({"$sample": {"size": args.sample_size}})

    builder = SchemaBuilder(schema_uri="https://json-schema.org/draft/2020-12/schema")
    builder.add_schema({"type": "object"})  # seed

    # Stream docs into the builder
    for doc in coll.aggregate(pipeline, allowDiskUse=True):
        # Convert MongoDB types to JSON-serializable types
        clean_doc = json.loads(json.dumps(doc, default=str))
        builder.add_object(clean_doc)

    schema = builder.to_schema()
    # Optional niceties for Mongo:
    # - mark _id as required if present often (genson won't automatically)
    # - coerce integer/number unions: left to post-processing if needed
    with open(args.out_json_schema, "w") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"Wrote JSON Schema → {args.out_json_schema}")


if __name__ == "__main__":
    main()
