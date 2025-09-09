#!/bin/bash

# Exit on any error, undefined variable, or pipe failure
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Do not source this script — run it directly with ./$(basename "$0")" >&2
  return 1
fi

show_help() {
  cat <<EOF
Usage: $(basename "$0") [--file FILE] [--count N]

Randomly selects N IDs from a text file and processes each.

Options:
  -f, --file FILE     Path to input file containing one ID per line (required)
  -n, --count N       Number of IDs to select randomly (default: 10)
  -h, --help          Show this help message and exit

Example:
  $(basename "$0") --file ids.txt --count 5
EOF
}

# Default values
input_file=""
count=10

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --file|-f)
      if [[ $# -lt 2 ]]; then
        echo "Error: --file requires a value" >&2
        show_help
        exit 1
      fi
      input_file="$2"
      shift 2
      ;;
    --count|-n)
      if [[ $# -lt 2 ]]; then
        echo "Error: --count requires a value" >&2
        show_help
        exit 1
      fi
      count="$2"
      shift 2
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      show_help
      exit 1
      ;;
  esac
done

# Validation
if [[ -z "$input_file" ]]; then
  echo "Error: --file is required" >&2
  show_help
  exit 1
fi

if [[ ! -f "$input_file" ]]; then
  echo "Error: file '$input_file' does not exist or is not a regular file" >&2
  exit 1
fi

if [[ ! -r "$input_file" ]]; then
  echo "Error: file '$input_file' is not readable" >&2
  exit 1
fi

if ! [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --count must be a positive integer (got: '$count')" >&2
  show_help
  exit 1
fi

# Check if file has enough lines
total_lines=$(wc -l < "$input_file")
if [[ "$count" -gt "$total_lines" ]]; then
  echo "Warning: requested $count IDs but file only has $total_lines lines. Using all available." >&2
  count="$total_lines"
fi

# Main logic
echo "Selecting $count random IDs from $input_file (total: $total_lines lines)" >&2
shuf "$input_file" | head -n "$count" | while IFS= read -r id; do
  if [[ -n "$id" ]]; then  # Skip empty lines
    echo "Processing ID: $id"
    # Add your actual processing logic here
  fi
done