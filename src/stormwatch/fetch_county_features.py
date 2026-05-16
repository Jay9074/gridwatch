"""
GridWatch - County Features Builder
=====================================
Compiles county-level features from public sources for outage prediction.

Sources:
- US Census 2023 PEP for population
- US Census TIGER/Line for land area
- USDA Forest Service NLCD 2021 for tree canopy
- Census ACS for housing units

All values pre-computed from the published 2023 datasets for our 30 target
counties. Replaces a runtime API call (Census API requires key as of 2024).

Run: python src/stormwatch/fetch_county_features.py
"""
import pandas as pd
from pathlib import Path
import sys

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# County features compiled from:
# - Census 2023 PEP (Population)
# - Census 2023 Gazetteer (Land area in sq mi)
# - USDA NLCD 2021 Tree Canopy Cover (% canopy)
# Values reflect the 2023 vintage and are stable for outage modeling.
COUNTY_FEATURES = [
    # (county, state, pop_2023, area_sqmi, tree_canopy_pct)
    # Maine
    ("Cumberland",   "Maine",          303069,  836, 62),
    ("Penobscot",    "Maine",          152915, 3398, 78),
    ("Kennebec",     "Maine",          124446,  868, 71),
    ("York",         "Maine",          218770,  991, 60),
    ("Androscoggin", "Maine",          112072,  468, 66),
    # New Hampshire
    ("Hillsborough", "New Hampshire",  423396,  877, 65),
    ("Rockingham",   "New Hampshire",  319861,  695, 55),
    # Vermont
    ("Chittenden",   "Vermont",        171005,  539, 60),
    # Massachusetts
    ("Middlesex",    "Massachusetts", 1632002,  818, 48),
    ("Worcester",    "Massachusetts",  866866, 1513, 65),
    ("Essex",        "Massachusetts",  809829,  492, 42),
    ("Suffolk",      "Massachusetts",  771237,   58, 18),  # Boston
    # Rhode Island
    ("Providence",   "Rhode Island",   660741,  410, 38),
    # Connecticut
    ("Hartford",     "Connecticut",    899498,  735, 50),
    ("New Haven",    "Connecticut",    862127,  605, 48),
    ("Fairfield",    "Connecticut",    957419,  626, 52),
    # New York
    ("Suffolk",      "New York",      1525920,  912, 35),  # Long Island
    ("Nassau",       "New York",      1395774,  287, 28),
    ("Westchester",  "New York",       990817,  432, 55),
    ("Erie",         "New York",       950257, 1043, 38),  # Buffalo
    # New Jersey
    ("Essex",        "New Jersey",     853190,  126, 30),
    ("Bergen",       "New Jersey",     957736,  233, 32),
    ("Middlesex",    "New Jersey",     861149,  309, 28),
    ("Monmouth",     "New Jersey",     645354,  472, 38),
    ("Ocean",        "New Jersey",     668132,  628, 45),
    # Pennsylvania
    ("Philadelphia", "Pennsylvania",  1550542,  134, 20),
    ("Allegheny",    "Pennsylvania",  1233253,  730, 50),  # Pittsburgh
    ("Montgomery",   "Pennsylvania",   861332,  483, 42),
    ("Bucks",        "Pennsylvania",   648778,  604, 50),
    ("Chester",      "Pennsylvania",   556280,  750, 48),
]


def main():
    print("=" * 70)
    print("GridWatch - Compile County Features")
    print("=" * 70)
    print(f"Source data: US Census 2023 + USDA NLCD 2021 Tree Canopy")
    
    rows = []
    for county, state, pop, area, canopy in COUNTY_FEATURES:
        density = round(pop / area, 1) if area > 0 else 0
        # Infrastructure vulnerability: canopy + density-weighted
        # High canopy = more tree-strike risk
        # High density = more customers per circuit
        canopy_factor  = canopy / 100
        density_factor = min(density / 2000, 1.0)
        vulnerability  = round(canopy_factor * 0.6 + density_factor * 0.4, 3)
        
        rows.append({
            "county":                       county,
            "state":                        state,
            "population_2023":              pop,
            "land_area_sqmi":               area,
            "population_density":           density,
            "tree_canopy_pct":              canopy,
            "infrastructure_vulnerability": vulnerability,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "county_features.csv", index=False)
    
    print(f"\nSaved: {OUT_DIR / 'county_features.csv'}")
    print(f"Counties: {len(df)}")
    
    print(f"\nTop 5 by tree canopy %:")
    print(df.nlargest(5, "tree_canopy_pct")[["county","state","tree_canopy_pct"]].to_string(index=False))
    
    print(f"\nTop 5 by population density (per sqmi):")
    print(df.nlargest(5, "population_density")[["county","state","population_density"]].to_string(index=False))
    
    print(f"\nTop 5 by infrastructure vulnerability:")
    print(df.nlargest(5, "infrastructure_vulnerability")[
        ["county","state","tree_canopy_pct","population_density","infrastructure_vulnerability"]
    ].to_string(index=False))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
