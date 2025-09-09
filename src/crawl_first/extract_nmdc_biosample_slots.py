#!/usr/bin/env python3
"""
Extract NMDC Biosample slots using LinkML SchemaView induction to include inherited slots.
"""

import json
from pathlib import Path

import click
from linkml_runtime.utils.schemaview import SchemaView


@click.command()
@click.option("--schema-path", required=True, help="Path to NMDC schema YAML file")
@click.option(
    "--output",
    default="nmdc_biosample_slots.json",
    help="Output JSON file with slots array",
)
def extract_slots(schema_path, output):
    """Extract NMDC Biosample slots including inherited ones using induction."""

    schema_path = Path(schema_path)

    if not schema_path.exists():
        click.echo(f"❌ Schema file not found: {schema_path}")
        raise click.Abort()

    click.echo(f"Loading schema from: {schema_path}")

    try:
        sv = SchemaView(str(schema_path))

        # Use induction to get complete class definition including inherited slots
        induced_class = sv.induced_class("Biosample")

        if not induced_class or not induced_class.attributes:
            click.echo("❌ No attributes found for induced Biosample class")
            raise click.Abort()

        slots = list(induced_class.attributes.keys())

        click.echo(f"✅ Found {len(slots)} NMDC Biosample slots (including inherited)")

        # Save as JSON
        with open(output, "w") as f:
            json.dump(sorted(slots), f, indent=2)

        click.echo(f"💾 Slots saved to: {output}")

    except Exception as e:
        click.echo(f"❌ Error using SchemaView induction: {e}")
        raise click.Abort()


if __name__ == "__main__":
    extract_slots()
