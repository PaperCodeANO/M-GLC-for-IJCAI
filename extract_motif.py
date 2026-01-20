import chembl_downloader
import pandas as pd
from dataset.motif_utils import bridge

# 1. Automatically download latest ChEMBL chemreps
chemreps_path = chembl_downloader.download_chemreps()
print(f"Downloaded ChEMBL chemreps to: {chemreps_path}")

# 2. Load with pandas
df = pd.read_csv(chemreps_path, sep="\t", compression="gzip")
df = df.sample(frac=0.2, random_state=42) 

# 3. Extract canonical SMILES
smiles_list = df['canonical_smiles'].dropna().unique().tolist()

# 4. (Optional) print or save
print(f"✅ Total unique SMILES: {len(smiles_list)}")
print("Sample:", smiles_list[:5])

from rdkit import Chem
from rdkit.Chem import BRICS
import re
from collections import defaultdict
from tqdm import tqdm

# def brics_motif(mol):
#     """
#     Given an RDKit Mol object, decompose it using BRICS,
#     and output a list of motifs where attachment points [*:n] are replaced with *_.
#     """
#     frags = BRICS.BRICSDecompose(mol)  # output: set of SMILES
#     clean_frags = []
#     for frag_smiles in frags:
#         unified = re.sub(r'\[\d+\*\]', '*_', frag_smiles)
#         clean_frags.append(unified)
#     return clean_frags


# def get_motif_dict(mol, motif_dict):
#     fragments = brics_motif(mol)
    
#     for i, frag in enumerate(fragments):
#        motif_dict[frag] += 1
#     return motif_dict 


total_molecules = 0
motif_dict = defaultdict(int)

for smiles in tqdm(smiles_list):
    if len(smiles) >= 200:
        continue
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        if mol is None:
            continue
        total_molecules += 1
        motif_dict = bridge(mol, motif_dict)
    
    except:
        continue

def filter_top_k_motifs(motif_dict, filter_number=500):
    sorted_items = sorted(motif_dict.items(), key=lambda x: -x[1])
    top_k_items = sorted_items[:filter_number]
    return dict(top_k_items)

filter_motif_dict = filter_top_k_motifs(motif_dict)

print(filter_motif_dict)

import json

def save_motif_dict(motif_dict, path="filter_motif_dict.json"):
    with open(path, "w") as f:
        json.dump(motif_dict, f, indent=2)
    print(f"✅ Saved to {path}")

save_motif_dict(filter_motif_dict, "filtered_motif.json")

