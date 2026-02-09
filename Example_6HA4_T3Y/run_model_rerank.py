#!/usr/bin/env python3
"""
Complete molecular docking analysis pipeline combining:
1. Complex generation
2. π-cation interaction analysis
3. Energy prediction and ranking
"""

import os
import sys
import logging
import glob
import csv
import math
import warnings
import pickle
import argparse
import tempfile
import shutil
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from io import StringIO
import numpy as np
import pandas as pd
import joblib
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit import RDLogger
from plip.structure.preparation import PDBComplex
from sklearn.preprocessing import StandardScaler

# ==================== CONFIGURATION ====================
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.info')

# Model paths
MODEL_DIR = Path(__file__).parent / "trained_models"
STANDARD_MODEL_PATH = MODEL_DIR / "non-ARG_energy_prediction_model.pkl"
ARG_MODEL_PATH = MODEL_DIR / "ARG_pi_interaction_energy_predictor.pkl"
RERANKER_MODEL_PATH = MODEL_DIR / "vina_failure_finetuned_best_model.pkl"

# ==================== CLASSES ====================
class ProteinSelect(Select):
    """Select only standard amino acids"""
    def accept_residue(self, residue):
        return is_aa(residue, standard=True)

# ==================== UTILITY FUNCTIONS ====================
def setup_logging():
    """Configure logging system"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    error_log_file = 'docking_analysis.log'
    fh = logging.FileHandler(error_log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger, error_log_file

logger, error_log_file = setup_logging()

def log_error(message):
    """Log error messages"""
    logger.error(message)
    with open(error_log_file, 'a') as f:
        f.write(f"{pd.Timestamp.now()}: {message}\n")

def print_logo():
    """Print program logo"""
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    PiCationDock                              ║
    ║                                                              ║
    ║                    Version 1.0.0                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

# ==================== STEP 1: COMPLEX GENERATION ====================
def clean_pdb(input_file, output_file):
    """Clean PDB file to contain only protein atoms"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', input_file)
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_file, ProteinSelect())
        logger.info(f"Generated clean protein: {output_file}")
        return output_file
    except Exception as e:
        log_error(f"Failed to clean PDB {input_file}: {str(e)}")
        return None

def find_protein_files(directory):
    """Find protein files in directory"""
    protein_files = glob.glob(os.path.join(directory, "*_protein.pdb"))
    if not protein_files:
        protein_files = glob.glob(os.path.join(directory, "*_protein_protonated.pdb"))
    
    clean_files = []
    for pf in protein_files:
        clean_file = pf.replace('_protein.pdb', '_clean.pdb').replace('_protein_protonated.pdb', '_clean.pdb')
        result = clean_pdb(pf, clean_file)
        if result:
            clean_files.append(result)
    
    return clean_files

def find_docked_files(directory):
    """Find docked ligand files"""
    return glob.glob(os.path.join(directory, "*_dock.sdf"))

def create_complex(protein_file, ligand_file, output_dir, pose_idx):
    """Create protein-ligand complex PDB"""
    try:
        # Load protein
        with open(protein_file, 'r') as f:
            protein_lines = f.readlines()
        
        # Load ligand pose
        supplier = Chem.SDMolSupplier(ligand_file)
        mol = supplier[pose_idx]
        if mol is None:
            return None
        
        # Generate output path
        prot_name = os.path.basename(protein_file).replace('_clean.pdb', '')
        lig_name = os.path.basename(ligand_file).replace('_dock.sdf', '')
        output_file = os.path.join(output_dir, f"{prot_name}_{lig_name}_complex_{pose_idx+1}.pdb")
        
        # Write complex
        with open(output_file, 'w') as f:
            f.writelines(protein_lines)
            f.write(Chem.MolToPDBBlock(mol))
            f.write("END\n")
        
        return output_file
    except Exception as e:
        log_error(f"Error creating complex {pose_idx+1}: {str(e)}")
        return None

def process_directory(directory):
    """Process all protein-ligand pairs in a directory"""
    protein_files = find_protein_files(directory)
    ligand_files = find_docked_files(directory)
    
    if not protein_files or not ligand_files:
        logger.warning(f"Skipping {directory}: missing protein or ligand files")
        return []
    
    complex_dirs = []
    for protein in protein_files:
        for ligand in ligand_files:
            # Create output directory
            prot_name = os.path.basename(protein).replace('_clean.pdb', '')
            lig_name = os.path.basename(ligand).replace('_dock.sdf', '')
            output_dir = os.path.join(directory, f"{prot_name}_{lig_name}_complexes")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate all poses
            supplier = Chem.SDMolSupplier(ligand)
            complexes = []
            for i in range(len(supplier)):
                complex_file = create_complex(protein, ligand, output_dir, i)
                if complex_file:
                    complexes.append(complex_file)
            
            if complexes:
                complex_dirs.append(output_dir)
    
    return complex_dirs

# ==================== STEP 2: INTERACTION ANALYSIS ====================
def analyze_interactions(pdb_file):
    """Analyze π-cation interactions in a complex"""
    try:
        pdbcomplex = PDBComplex()
        pdbcomplex.load_pdb(str(pdb_file))
        pdbcomplex.analyze()
        
        interactions = []
        for bs_id, interaction_set in pdbcomplex.interaction_sets.items():
            for pic in interaction_set.pication_laro:
                # Extract interaction features
                interaction = {
                    'complex': os.path.basename(pdb_file),
                    'protein_res': f"{pic.restype}-{pic.resnr}-{pic.reschain}",
                    'ligand': f"{pic.restype_l}-{pic.resnr_l}-{pic.reschain_l}",
                    'distance': round(float(pic.distance), 2),
                    'offset': round(float(pic.offset), 2),
                    'angle': round(float(pic.angle), 2),
                    'is_arg': (pic.restype.strip().upper() == 'ARG')
                }
                interactions.append(interaction)
        
        return interactions
    except Exception as e:
        log_error(f"Error analyzing {pdb_file}: {str(e)}")
        return []

def analyze_all_complexes(complex_dirs):
    """Analyze all complexes in parallel"""
    all_pdbs = []
    for d in complex_dirs:
        all_pdbs.extend(glob.glob(os.path.join(d, "*_complex_*.pdb")))
    
    if not all_pdbs:
        logger.error("No complex PDB files found!")
        return None
    
    # Parallel processing
    with Pool(processes=min(cpu_count(), 8)) as pool:
        results = []
        with tqdm(total=len(all_pdbs), desc="Analyzing complexes") as pbar:
            for res in pool.imap_unordered(analyze_interactions, all_pdbs):
                if res:
                    results.extend(res)
                pbar.update(1)
    
    if not results:
        logger.error("No interactions found!")
        return None
    
    # Save to temp CSV
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        writer = csv.DictWriter(tmp, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        return tmp.name

# ==================== STEP 3: MODEL PREDICTION ====================
def load_model(model_path):
    """Load a trained model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        log_error(f"Failed to load model {model_path}: {str(e)}")
        return None

def predict_energies(interactions_csv):
    """Predict interaction energies"""
    df = pd.read_csv(interactions_csv)
    if df.empty:
        logger.error("No interactions to predict!")
        return None
    
    # Load models
    arg_model = load_model(ARG_MODEL_PATH)
    std_model = load_model(STANDARD_MODEL_PATH)
    reranker_model = load_model(RERANKER_MODEL_PATH)
    
    if not all([arg_model, std_model, reranker_model]):
        logger.error("Missing required models!")
        return None
    
    # Predict ARG interactions
    arg_mask = df['is_arg'] == True
    if arg_mask.any():
        df_arg = df[arg_mask].copy()
        # Feature engineering for ARG
        df_arg['inv_distance'] = 1.0 / df_arg['distance']
        df_arg['distance_sq'] = df_arg['distance'] ** 2
        # Predict (example features - adjust based on actual model)
        X_arg = df_arg[['distance', 'offset', 'angle', 'inv_distance', 'distance_sq']]
        df_arg['predicted_energy'] = arg_model.predict(X_arg)
    
    # Predict non-ARG interactions
    other_mask = df['is_arg'] == False
    if other_mask.any():
        df_other = df[other_mask].copy()
        # Predict (example features - adjust based on actual model)
        X_other = df_other[['distance', 'offset', 'angle']]
        df_other['predicted_energy'] = std_model.predict(X_other)
    
    # Combine predictions
    df_pred = pd.concat([df_arg, df_other], ignore_index=True)
    
    # Rerank predictions
    df_pred['rank'] = df_pred.groupby('complex')['predicted_energy'].rank()
    
    # Save final results
    output_file = "docking_results.csv"
    df_pred.to_csv(output_file, index=False)
    return output_file

# ==================== MAIN EXECUTION ====================
def main(base_dir):
    """Run complete analysis pipeline"""
    print_logo()
    
    # Step 1: Generate complexes
    logger.info("\n" + "="*50)
    logger.info("STEP 1: GENERATING COMPLEXES")
    logger.info("="*50)
    complex_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            full_path = os.path.join(root, d)
            dir_complexes = process_directory(full_path)
            complex_dirs.extend(dir_complexes)
    
    if not complex_dirs:
        logger.error("No complexes generated!")
        return 1
    
    # Step 2: Analyze interactions
    logger.info("\n" + "="*50)
    logger.info("STEP 2: ANALYZING INTERACTIONS")
    logger.info("="*50)
    interactions_csv = analyze_all_complexes(complex_dirs)
    if not interactions_csv:
        return 1
    
    # Step 3: Predict and rank
    logger.info("\n" + "="*50)
    logger.info("STEP 3: PREDICTING ENERGIES")
    logger.info("="*50)
    results_file = predict_energies(interactions_csv)
    if not results_file:
        return 1
    
    logger.info(f"\nAnalysis complete! Results saved to: {results_file}")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Molecular Docking Analysis Pipeline')
    parser.add_argument('base_dir', help='Directory containing docking results')
    args = parser.parse_args()
    
    sys.exit(main(args.base_dir))

