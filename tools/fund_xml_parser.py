import os
import xml.etree.ElementTree as ET
from typing import List, Dict

def extract_exposure_from_xml(directory: str, target_company: str) -> List[Dict]:
    """
    Recursively scans all XML files in a directory and subdirectories
    to extract exposure to the target company.
    """
    results = []
    for root_dir, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".xml"):
                continue

            path = os.path.join(root_dir, filename)
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                fund_name = root.attrib.get("Fondnamn") or filename

                for innehav in root.iter("Innehav"):
                    bolag = innehav.findtext("Bolagsnamn", "").lower()
                    if target_company.lower() in bolag:
                        isin = innehav.findtext("ISIN", "N/A")
                        percent = innehav.findtext("AndelAvFond", "0").replace(",", ".")
                        results.append({
                            "fund": fund_name,
                            "isin": isin,
                            "company": bolag.title(),
                            "exposure_pct": float(percent),
                            "source": path
                        })
            except Exception as e:
                results.append({"error": f"{filename}: {str(e)}"})

    return results

