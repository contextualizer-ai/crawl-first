#!/usr/bin/env python3
"""
crawl-first: Deterministic biosample enrichment for LLM-ready data preparation.

Systematically follows discoverable links from NMDC biosample records to gather
environmental, geospatial, weather, publication, and ontological data.
"""

import random
from pathlib import Path
from typing import List, Optional

import click
import yaml

from .biosample import analyze_biosample
from .logging_utils import setup_logging
from .yaml_utils import NoRefsDumper


def load_biosample_ids(file_path: str, sample_size: Optional[int] = None) -> List[str]:
    """Load biosample IDs from file with optional sampling."""
    try:
        with open(file_path, "r") as f:
            all_ids = [line.strip() for line in f if line.strip()]

        if sample_size is not None and sample_size < len(all_ids):
            return random.sample(all_ids, sample_size)
        return all_ids
    except Exception:
        return []


@click.command()
@click.option("--biosample-id", help="Single NMDC biosample ID")
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    help="File with biosample IDs (one per line)",
)
@click.option("--sample-size", type=int, help="Random sample size from input file")
@click.option(
    "--output-file",
    type=click.Path(),
    help="Output YAML file (single file for all results)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Output directory (individual YAML files per biosample)",
)
@click.option(
    "--email",
    required=True,
    help="Email address for API requests (required for full text fetching)",
)
@click.option(
    "--search-radius",
    type=int,
    default=1000,
    help="Search radius in meters for geospatial features (default: 1000)",
)
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.option(
    "--capture-output/--no-capture-output",
    default=True,
    help="Capture stdout/stderr to log file (default: True)",
)
def main(
    biosample_id: str,
    input_file: str,
    sample_size: int,
    output_file: str,
    output_dir: str,
    email: str,
    search_radius: int,
    verbose: bool,
    capture_output: bool,
) -> None:
    """crawl-first: Deterministic biosample enrichment for LLM-ready data preparation."""

    # Setup logging
    logger, output_capture = setup_logging(verbose, capture_output)

    # Validate input options
    if input_file and biosample_id:
        logger.error("Cannot specify both --biosample-id and --input-file")
        raise click.ClickException(
            "Cannot specify both --biosample-id and --input-file"
        )
    elif not input_file and not biosample_id:
        logger.error("Must specify either --biosample-id or --input-file")
        raise click.ClickException("Must specify either --biosample-id or --input-file")

    # Validate output options
    if output_file and output_dir:
        logger.error("Cannot specify both --output-file and --output-dir")
        raise click.ClickException("Cannot specify both --output-file and --output-dir")
    elif not output_file and not output_dir:
        logger.error("Must specify either --output-file or --output-dir")
        raise click.ClickException("Must specify either --output-file or --output-dir")

    # Get biosample IDs
    if input_file:
        logger.info(f"Loading biosample IDs from {input_file}")
        biosample_ids = load_biosample_ids(input_file, sample_size)
        if not biosample_ids:
            logger.error("No biosample IDs to process")
            raise click.ClickException("No biosample IDs to process")
        logger.info(f"Loaded {len(biosample_ids)} biosample IDs for processing")
    else:
        biosample_ids = [biosample_id]
        logger.info(f"Processing single biosample: {biosample_id}")

    # Process biosamples
    results = {}
    failed = []

    # Create output directory if using directory mode
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_path}")

    for i, bid in enumerate(biosample_ids):
        logger.info(f"Processing {i+1}/{len(biosample_ids)}: {bid}")

        result = analyze_biosample(bid, email, search_radius)
        if result:
            if output_dir:
                # Write individual YAML file immediately
                filename = bid.replace(":", "_") + ".yaml"
                file_path = output_path / filename
                try:
                    with open(file_path, "w") as f:
                        yaml.dump(
                            result,
                            f,
                            Dumper=NoRefsDumper,
                            default_flow_style=False,
                            sort_keys=False,
                            indent=2,
                        )
                    logger.debug(f"Saved: {file_path}")
                except Exception as e:
                    logger.error(f"Error saving {file_path}: {e}")
                    failed.append(bid)
                    continue
            else:
                # Collect for batch output (single file mode)
                results[bid] = result
        else:
            logger.warning(f"Failed to analyze biosample: {bid}")
            failed.append(bid)

    # Report summary
    if len(biosample_ids) > 1:
        if output_dir:
            successful_count = len(biosample_ids) - len(failed)
            logger.info(
                f"Processed {successful_count}/{len(biosample_ids)} biosamples successfully"
            )
            logger.info(f"Files saved to: {output_dir}")
        else:
            logger.info(
                f"Processed {len(results)}/{len(biosample_ids)} biosamples successfully"
            )

        if failed:
            logger.warning(f"Failed biosamples: {', '.join(failed)}")

    # Output results (only for single file mode since directory mode saves immediately)
    if output_file and results:
        output_data = list(results.values())[0] if len(results) == 1 else results
        with open(output_file, "w") as f:
            yaml.dump(
                output_data,
                f,
                Dumper=NoRefsDumper,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
            )
        logger.info(f"Data saved to: {output_file}")
    elif results and not output_file and not output_dir:
        # Simple display for single biosample
        if len(results) == 1:
            result = list(results.values())[0]
            asserted = result["asserted"]
            inferred = result["inferred"]

            logger.info(f"Biosample: {asserted.get('id')}")

            # Show coordinate sources
            coord_sources = inferred.get("coordinate_sources", {})
            if "from_asserted_coords" in coord_sources:
                coords = coord_sources["from_asserted_coords"]["coordinates"]
                logger.info(
                    f"Asserted coordinates: {coords['latitude']}, {coords['longitude']}"
                )

            if "from_geo_loc_name" in coord_sources:
                coords = coord_sources["from_geo_loc_name"]["coordinates"]
                source = coord_sources["from_geo_loc_name"]["source"]
                logger.info(
                    f"Geocoded coordinates: {coords['latitude']}, {coords['longitude']} ({source})"
                )

            # Show soil analysis
            for soil_key in ["soil_from_asserted_coords", "soil_from_geo_loc_name"]:
                if soil_key in inferred:
                    soil_type = inferred[soil_key].get("soil_type")
                    if soil_type:
                        source = (
                            "asserted coords"
                            if "asserted" in soil_key
                            else "geo_loc_name"
                        )
                        logger.info(f"Soil ({source}): {soil_type}")

            # Show publication info
            if "publication_analysis" in inferred:
                pub_data = inferred["publication_analysis"]
                logger.info(
                    f"Study DOIs: {pub_data.get('total_dois', 0)} total, {pub_data.get('publication_doi_count', 0)} publications"
                )

            # Show geospatial info from both sources
            for geo_key in [
                "geospatial_from_asserted_coords",
                "geospatial_from_geo_loc_name",
            ]:
                if geo_key in inferred:
                    geo_data = inferred[geo_key]
                    source = (
                        "asserted coords" if "asserted" in geo_key else "geo_loc_name"
                    )

                    elevation = geo_data.get("elevation")
                    if elevation:
                        logger.info(f"Elevation ({source}): {elevation.get('meters')}m")

                    place_info = geo_data.get("place_info", {})
                    if place_info.get("display_name"):
                        logger.info(
                            f"Location ({source}): {place_info['display_name']}"
                        )

                    env_summary = geo_data.get("environmental_summary", {})
                    total_features = env_summary.get("total_environmental_features", 0)
                    if total_features > 0:
                        logger.info(
                            f"Environmental features ({source}): {total_features} within {search_radius}m"
                        )

    if not results:
        logger.error("No data processed successfully")

    # Cleanup output capture
    if output_capture:
        output_capture.stop()


if __name__ == "__main__":
    main()
