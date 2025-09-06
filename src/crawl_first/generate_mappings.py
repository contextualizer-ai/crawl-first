"""
Generate ENVO ontology mappings for geospatial classification systems.

Supports both deterministic and AI-assisted mapping generation:
- Deterministic: Use known official mappings where available
- AI-assisted: Use Claude Code or OLS API to find best ENVO matches
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class ENVOMappingGenerator:
    """Generate mappings from classification systems to ENVO ontology terms."""
    
    def __init__(self, mappings_dir: str = "mappings"):
        self.mappings_dir = Path(mappings_dir)
        self.mappings_dir.mkdir(exist_ok=True)
        
    def save_mapping(self, name: str, mapping: Dict[str, Any]) -> None:
        """Save mapping to JSON file."""
        output_file = self.mappings_dir / f"{name}.json"
        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=2, sort_keys=True)
        print(f"Saved mapping to {output_file}")
        
    def generate_soilgrids_fao_mapping_deterministic(self) -> Dict[str, Any]:
        """
        Generate FAO soil type to ENVO mapping using known deterministic mappings.
        
        These mappings are based on official FAO classifications and established
        ENVO ontology terms for soil types.
        """
        # Known FAO soil types with their ENVO equivalents
        # These are deterministic mappings based on official classifications
        fao_to_envo = {
            "Acrisols": {
                "id": "ENVO:00002263", 
                "label": "acrisol", 
                "term": "Acrisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Albeluvisols": {
                "id": "ENVO:00002275", 
                "label": "albeluvisol", 
                "term": "Albeluvisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Alisols": {
                "id": "ENVO:00002264", 
                "label": "alisol", 
                "term": "Alisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Andosols": {
                "id": "ENVO:00002265", 
                "label": "andosol", 
                "term": "Andosol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Arenosols": {
                "id": "ENVO:00002266", 
                "label": "arenosol", 
                "term": "Arenosol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Calcisols": {
                "id": "ENVO:00002267", 
                "label": "calcisol", 
                "term": "Calcisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Cambisols": {
                "id": "ENVO:00002268", 
                "label": "cambisol", 
                "term": "Cambisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Chernozems": {
                "id": "ENVO:00002269", 
                "label": "chernozem", 
                "term": "Chernozem",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Cryosols": {
                "id": "ENVO:00002270", 
                "label": "cryosol", 
                "term": "Cryosol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Durisols": {
                "id": "ENVO:00002271", 
                "label": "durisol", 
                "term": "Durisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Ferralsols": {
                "id": "ENVO:00002272", 
                "label": "ferralsol", 
                "term": "Ferralsol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Fluvisols": {
                "id": "ENVO:00002273", 
                "label": "fluvisol", 
                "term": "Fluvisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Gleysols": {
                "id": "ENVO:00002274", 
                "label": "gleysol", 
                "term": "Gleysol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Histosols": {
                "id": "ENVO:00002276", 
                "label": "histosol", 
                "term": "Histosol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Kastanozems": {
                "id": "ENVO:00002277", 
                "label": "kastanozem", 
                "term": "Kastanozem",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Leptosols": {
                "id": "ENVO:00002278", 
                "label": "leptosol", 
                "term": "Leptosol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Lixisols": {
                "id": "ENVO:00002279", 
                "label": "lixisol", 
                "term": "Lixisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Luvisols": {
                "id": "ENVO:00002280", 
                "label": "luvisol", 
                "term": "Luvisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Nitisols": {
                "id": "ENVO:00002281", 
                "label": "nitisol", 
                "term": "Nitisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Phaeozems": {
                "id": "ENVO:00002282", 
                "label": "phaeozem", 
                "term": "Phaeozem",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Planosols": {
                "id": "ENVO:00002283", 
                "label": "planosol", 
                "term": "Planosol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Plinthosols": {
                "id": "ENVO:00002284", 
                "label": "plinthosol", 
                "term": "Plinthosol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Podzols": {
                "id": "ENVO:00002285", 
                "label": "podzol", 
                "term": "Podzol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Regosols": {
                "id": "ENVO:00002286", 
                "label": "regosol", 
                "term": "Regosol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Solonchaks": {
                "id": "ENVO:00002287", 
                "label": "solonchak", 
                "term": "Solonchak",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Solonetz": {
                "id": "ENVO:00002288", 
                "label": "solonetz", 
                "term": "Solonetz",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Stagnosols": {
                "id": "ENVO:00002289", 
                "label": "stagnosol", 
                "term": "Stagnosol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Umbrisols": {
                "id": "ENVO:00002290", 
                "label": "umbrisol", 
                "term": "Umbrisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            },
            "Vertisols": {
                "id": "ENVO:00002291", 
                "label": "vertisol", 
                "term": "Vertisol",
                "confidence": "deterministic",
                "source": "FAO World Reference Base"
            }
        }
        
        return fao_to_envo
    
    def generate_esa_worldcover_mapping_with_ai(self) -> Dict[str, Any]:
        """
        Generate ESA WorldCover to ENVO mapping using AI assistance.
        
        Uses Claude Code to analyze WorldCover classes and find best ENVO matches.
        """
        # ESA WorldCover classes with descriptions
        worldcover_classes = {
            "10": {"label": "Tree cover", "description": "All types of forest and woodland"},
            "20": {"label": "Shrubland", "description": "Shrub and scrub habitats"}, 
            "30": {"label": "Grassland", "description": "Natural and semi-natural grasslands"},
            "40": {"label": "Cropland", "description": "Agricultural and cultivated areas"},
            "50": {"label": "Built-up", "description": "Urban and built-up areas"},
            "60": {"label": "Bare/sparse vegetation", "description": "Areas with little to no vegetation"},
            "70": {"label": "Snow and Ice", "description": "Permanent snow and ice"},
            "80": {"label": "Permanent water bodies", "description": "Rivers, lakes, coastal waters"},
            "90": {"label": "Herbaceous wetland", "description": "Wetland areas with herbaceous cover"},
            "95": {"label": "Mangroves", "description": "Mangrove ecosystems"},
            "100": {"label": "Moss and lichen", "description": "Moss and lichen dominated areas"}
        }
        
        mapping = {}
        
        # Use AI to find best ENVO matches
        for class_id, class_info in worldcover_classes.items():
            envo_match = self._find_envo_match_with_ai(
                class_info["label"], 
                class_info["description"],
                "land cover"
            )
            
            if envo_match:
                mapping[class_id] = {
                    **envo_match,
                    "term": class_info["label"],
                    "confidence": "ai_assisted",
                    "source": "ESA WorldCover 10m v200"
                }
                
        return mapping
    
    def generate_nlcd_mapping_with_ai(self) -> Dict[str, Any]:
        """Generate NLCD to ENVO mapping using AI assistance."""
        # NLCD classes 
        nlcd_classes = {
            "11": {"label": "Open Water", "description": "All areas of open water with less than 25% vegetation"},
            "12": {"label": "Perennial Ice/Snow", "description": "All areas with perennial ice and snow"},
            "21": {"label": "Developed, Open Space", "description": "Mixture of vegetation and constructed materials, mostly vegetation"},
            "22": {"label": "Developed, Low Intensity", "description": "Mixture of constructed materials and vegetation, 20-49% constructed"},
            "23": {"label": "Developed, Medium Intensity", "description": "Mixture of constructed materials and vegetation, 50-79% constructed"},
            "24": {"label": "Developed, High Intensity", "description": "Highly developed areas, 80-100% constructed materials"},
            "31": {"label": "Barren Land", "description": "Rock, sand, clay, or other earthen material with little to no vegetation"},
            "41": {"label": "Deciduous Forest", "description": "Dominated by trees with more than 75% deciduous species"},
            "42": {"label": "Evergreen Forest", "description": "Dominated by trees with more than 75% evergreen species"},
            "43": {"label": "Mixed Forest", "description": "Dominated by trees with neither deciduous nor evergreen species > 75%"},
            "51": {"label": "Dwarf Scrub", "description": "Alaska only - shrubs less than 20 cm tall"},
            "52": {"label": "Shrub/Scrub", "description": "Dominated by shrubs less than 5 meters tall"},
            "71": {"label": "Grassland/Herbaceous", "description": "Dominated by gramanoid or herbaceous vegetation"},
            "72": {"label": "Sedge/Herbaceous", "description": "Alaska only - dominated by sedges and forbs"},
            "73": {"label": "Lichens", "description": "Alaska only - dominated by fruticose or foliose lichens"},
            "74": {"label": "Moss", "description": "Alaska only - dominated by mosses"},
            "81": {"label": "Pasture/Hay", "description": "Grasses, legumes, or other herbaceous plants for livestock grazing"},
            "82": {"label": "Cultivated Crops", "description": "Areas used for production of annual crops"},
            "90": {"label": "Woody Wetlands", "description": "Forest or shrub wetlands where water is present"},
            "95": {"label": "Emergent Herbaceous Wetlands", "description": "Perennial herbaceous wetlands"}
        }
        
        mapping = {}
        
        for class_id, class_info in nlcd_classes.items():
            envo_match = self._find_envo_match_with_ai(
                class_info["label"], 
                class_info["description"],
                "land cover"
            )
            
            if envo_match:
                mapping[class_id] = {
                    **envo_match,
                    "term": class_info["label"],
                    "confidence": "ai_assisted",
                    "source": "USGS NLCD"
                }
                
        return mapping
    
    def _find_envo_match_with_ai(self, term: str, description: str, context: str) -> Optional[Dict[str, Any]]:
        """Find best ENVO match using AI assistance."""
        prompt = f"""
        Find the best ENVO (Environmental Ontology) term match for this {context} classification:
        
        Term: {term}
        Description: {description}
        
        Please provide:
        1. The most appropriate ENVO ID (format: ENVO:XXXXXXXX)
        2. The ENVO term label
        3. A brief explanation of why this is the best match
        
        Focus on finding the most specific, accurate ENVO term that represents this classification.
        Return only the ENVO ID, label, and explanation in this format:
        
        ID: ENVO:XXXXXXXX
        Label: [term label]
        Explanation: [brief explanation]
        """
        
        try:
            # Try using Claude Code CLI if available
            result = subprocess.run([
                'claude', '--print', '--no-mcp'
            ], 
            input=prompt, 
            text=True, 
            capture_output=True, 
            timeout=60
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                return self._parse_ai_response(response)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        # Fallback to OLS API search
        return self._find_envo_match_with_ols(term, description)
    
    def _find_envo_match_with_ols(self, term: str, description: str) -> Optional[Dict[str, Any]]:
        """Find ENVO match using OLS API as fallback."""
        try:
            # Search ENVO ontology using OLS API
            url = "https://www.ebi.ac.uk/ols/api/search"
            params = {
                'q': term,
                'ontology': 'envo',
                'rows': 3,
                'exact': 'false'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            docs = data.get('response', {}).get('docs', [])
            
            if docs:
                # Take the best match
                best_match = docs[0]
                
                return {
                    "id": best_match.get('obo_id', ''),
                    "label": best_match.get('label', ''),
                    "explanation": f"Best match from OLS API search for '{term}'"
                }
                
        except requests.RequestException:
            pass
            
        return None
    
    def _parse_ai_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse AI response to extract ENVO ID, label, and explanation."""
        lines = response.split('\n')
        envo_data = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('ID:'):
                envo_data['id'] = line.replace('ID:', '').strip()
            elif line.startswith('Label:'):
                envo_data['label'] = line.replace('Label:', '').strip()
            elif line.startswith('Explanation:'):
                envo_data['explanation'] = line.replace('Explanation:', '').strip()
                
        if envo_data.get('id') and envo_data.get('label'):
            return envo_data
            
        return None
    
    def generate_all_mappings(self, use_ai: bool = True) -> None:
        """Generate all mapping files."""
        print("Generating ENVO ontology mappings...")
        
        # Generate SoilGrids FAO mapping (deterministic)
        print("\n1. Generating SoilGrids FAO to ENVO mapping (deterministic)...")
        soilgrids_mapping = self.generate_soilgrids_fao_mapping_deterministic()
        self.save_mapping("soilgrids_fao_to_envo", soilgrids_mapping)
        
        if use_ai:
            # Generate ESA WorldCover mapping (AI-assisted)
            print("\n2. Generating ESA WorldCover to ENVO mapping (AI-assisted)...")
            esa_mapping = self.generate_esa_worldcover_mapping_with_ai()
            self.save_mapping("esa_worldcover_to_envo", esa_mapping)
            
            # Generate NLCD mapping (AI-assisted)  
            print("\n3. Generating NLCD to ENVO mapping (AI-assisted)...")
            nlcd_mapping = self.generate_nlcd_mapping_with_ai()
            self.save_mapping("nlcd_to_envo", nlcd_mapping)
        else:
            print("\n2-3. Skipping AI-assisted mappings (use_ai=False)")
            
        print("\nMapping generation complete!")


def main():
    """Main function to generate all mappings."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ENVO ontology mappings")
    parser.add_argument('--no-ai', action='store_true', help='Skip AI-assisted mappings')
    parser.add_argument('--mappings-dir', default='../mappings', help='Output directory for mappings')
    
    args = parser.parse_args()
    
    generator = ENVOMappingGenerator(args.mappings_dir)
    generator.generate_all_mappings(use_ai=not args.no_ai)


if __name__ == "__main__":
    main()