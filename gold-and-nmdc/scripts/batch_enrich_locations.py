#!/usr/bin/env python3
"""
Batch location enrichment from JSON input file.
"""

import click
import json
from pathlib import Path
from geospatial_enrichment import enrich_location

@click.command()
@click.option('--input', 'input_file', type=click.Path(exists=True, path_type=Path), required=True, 
              help='Input JSON file with locations array')
@click.option('--output', 'output_file', type=click.Path(path_type=Path), required=True,
              help='Output JSON file for enriched results')
@click.option('--verbose', is_flag=True, help='Show enrichment progress')
def main(input_file, output_file, verbose):
    """Batch location enrichment from JSON input file."""
    
    # Load input locations
    try:
        with open(input_file, 'r') as f:
            locations = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        click.echo(f"Error reading input file: {e}", err=True)
        raise click.Abort()
    
    if not isinstance(locations, list):
        click.echo("Error: Input file must contain a JSON array of locations", err=True)
        raise click.Abort()
    
    if verbose:
        click.echo(f"Loaded {len(locations)} locations from {input_file}")
        click.echo()
    
    results = {}
    
    for i, location in enumerate(locations, 1):
        if not isinstance(location, dict):
            click.echo(f"Skipping invalid location {i}: not a dict", err=True)
            continue
            
        name = location.get('name', f'Location_{i}')
        
        if 'lat' not in location or 'lon' not in location:
            click.echo(f"Skipping {name}: missing lat/lon coordinates", err=True)
            continue
            
        lat = location['lat']
        lon = location['lon']
        date = location.get('date')
        
        if verbose:
            click.echo(f"{i}. Enriching {name}")
            click.echo(f"   Coordinates: {lat}, {lon}")
            if date:
                click.echo(f"   Date: {date}")
        
        try:
            result = enrich_location(lat, lon, date)
            results[name] = result
            
            if verbose:
                # Show key results
                if result.get("elevation", {}).get("success"):
                    elev = result["elevation"]["elevation_meters"]
                    click.echo(f"   • Elevation: {elev}m")
                
                if result.get("weather", {}).get("success"):
                    weather = result["weather"]["weather_data"]
                    if "temperature_2m_mean" in weather:
                        temp = weather["temperature_2m_mean"]
                        click.echo(f"   • Temperature: {temp}°C")
                
                enrichments = result.get("enrichment_summary", {}).get("total_enrichments", 0)
                click.echo(f"   • Success: {enrichments}/4 APIs")
                click.echo()
                
        except Exception as e:
            click.echo(f"   Error enriching {name}: {e}", err=True)
            results[name] = {"error": str(e)}
    
    # Save results
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if verbose:
            click.echo(f"Results saved to {output_file}")
            
    except OSError as e:
        click.echo(f"Error writing output file: {e}", err=True)
        raise click.Abort()

if __name__ == "__main__":
    main()