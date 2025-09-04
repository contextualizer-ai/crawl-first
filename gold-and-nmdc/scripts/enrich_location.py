#!/usr/bin/env python3
"""
Single location enrichment using working geospatial APIs.
"""

import click
import json
from pathlib import Path
from geospatial_enrichment import enrich_location

@click.command()
@click.option('--lat', type=float, required=True, help='Latitude coordinate')
@click.option('--lon', type=float, required=True, help='Longitude coordinate')
@click.option('--date', type=str, help='Collection date in YYYY-MM-DD format')
@click.option('--output', type=click.Path(path_type=Path), help='Output JSON file (default: stdout)')
@click.option('--verbose', is_flag=True, help='Show enrichment progress')
def main(lat, lon, date, output, verbose):
    """Single location enrichment using working geospatial APIs."""
    
    if verbose:
        click.echo(f"Enriching location: {lat}, {lon}")
        if date:
            click.echo(f"Collection date: {date}")
        click.echo()
    
    try:
        result = enrich_location(lat, lon, date)
        
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            if verbose:
                click.echo(f"Results written to {output}")
        else:
            click.echo(json.dumps(result, indent=2))
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

if __name__ == "__main__":
    main()