import logging 
import os
import sys
import glob
import csv
import math
import numpy as np
import pandas as pd
import joblib
from decimal import Decimal
from multiprocessing import Pool, cpu_count, set_start_method
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit import RDLogger
from plip.structure.preparation import PDBComplex
from pathlib import Path
from tqdm import tqdm
import tempfile
import shutil
import warnings
from functools import partial

# Configure logging FIRST before using it anywhere
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Filter warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.info')

# ==================== USED MODEL PATHS - UPDATED WITH ROBUST SEARCH ====================
def find_project_root(start_path=None):
    """
    Find the project root directory containing 'pication-main' by searching upwards.
    Returns the path to pication-main directory.
    """
    if start_path is None:
        start_path = Path.cwd()
    
    current_path = Path(start_path).resolve()
    
    # Search upwards through parent directories
    for parent in [current_path] + list(current_path.parents):
        # Check if 'pication-main' directory exists in this parent
        pication_main = parent / "pication-main"
        if pication_main.exists() and pication_main.is_dir():
            return pication_main
    
    # If not found, try alternative naming patterns
    for parent in [current_path] + list(current_path.parents):
        # Look for directories containing 'pication' and 'main'
        for item in parent.iterdir():
            if item.is_dir() and ('pication' in item.name.lower() and 'main' in item.name.lower()):
                return item
    
    return None

def get_model_paths():
    """Dynamically locate model files by finding pication-main directory regardless of current depth"""
    
    # Find the project root (pication-main directory)
    project_root = find_project_root()
    
    if project_root is None:
        # Fallback: try to construct path based on common patterns
        current_dir = Path.cwd()
        possible_roots = [
            current_dir.parent.parent.parent,  # Go up 3 levels
            current_dir.parent.parent,          # Go up 2 levels  
            current_dir.parent,                 # Go up 1 level
            current_dir,                        # Current level
        ]
        
        for root in possible_roots:
            potential_pication = root / "pication-main"
            if potential_pication.exists():
                project_root = potential_pication
                break
    
    if project_root is None:
        logger.error("❌ Could not locate pication-main directory!")
        logger.error(f"Current working directory: {Path.cwd()}")
        logger.error("Please ensure you're running from within the pication project structure")
        logger.error("Expected structure: .../pication-main/trained_models/")
        # Return default paths but they will fail later
        default_location = Path.cwd().parent / "trained_models"
        return (
            default_location / "non-ARG_energy_prediction_model.pkl",
            default_location / "ARG_pi_interaction_energy_predictor.pkl"
        )
    
    # Construct path to trained_models directory
    trained_models_dir = project_root / "trained_models"
    
    if not trained_models_dir.exists() or not trained_models_dir.is_dir():
        logger.error(f"❌ trained_models directory not found at: {trained_models_dir}")
        logger.error(f"Project root found at: {project_root}")
        # Return paths that will fail gracefully
        return (
            trained_models_dir / "non-ARG_energy_prediction_model.pkl",
            trained_models_dir / "ARG_pi_interaction_energy_predictor.pkl"
        )
    
    logger.info(f"✅ Found project root: {project_root}")
    logger.info(f"✅ Found trained_models directory: {trained_models_dir}")
    
    # Check for model files
    standard_model = trained_models_dir / "non-ARG_energy_prediction_model.pkl"
    arg_model = trained_models_dir / "ARG_pi_interaction_energy_predictor.pkl"
    
    if standard_model.exists() and arg_model.exists():
        logger.info(f"✅ Found both model files:")
        logger.info(f"   Standard model: {standard_model}")
        logger.info(f"   ARG model: {arg_model}")
        return standard_model, arg_model
    else:
        # List what's actually in the directory
        available_files = list(trained_models_dir.glob("*.pkl"))
        logger.warning(f"⚠️  Model files not found in {trained_models_dir}")
        logger.info(f"Available .pkl files: {[f.name for f in available_files]}")
        
        # Try to find similar named files
        standard_candidates = list(trained_models_dir.glob("*non-ARG*pkl"))
        arg_candidates = list(trained_models_dir.glob("*ARG*pkl"))
        
        if standard_candidates and arg_candidates:
            logger.info(f"Using candidate files:")
            logger.info(f"   Standard: {standard_candidates[0]}")
            logger.info(f"   ARG: {arg_candidates[0]}")
            return standard_candidates[0], arg_candidates[0]
        elif available_files:
            # Use first two .pkl files if we can't find the specific ones
            if len(available_files) >= 2:
                logger.warning(f"Using first two .pkl files as fallback:")
                logger.info(f"   Standard: {available_files[0]}")
                logger.info(f"   ARG: {available_files[1]}")
                return available_files[0], available_files[1]
            elif len(available_files) == 1:
                logger.warning(f"Only one .pkl file found, using it for both models:")
                logger.info(f"   Using: {available_files[0]}")
                return available_files[0], available_files[0]
        
        # Return the expected paths even if they don't exist (will fail gracefully later)
        return standard_model, arg_model

# Initialize model paths
STANDARD_MODEL_PATH, ARG_MODEL_PATH = get_model_paths()

# Additional diagnostic function to check model files
def diagnose_model_file(file_path):
    """Diagnose what's actually in the model file"""
    try:
        logger.info(f"🔍 Diagnosing model file: {file_path}")
        
        # Check if file exists first
        if not file_path.exists():
            logger.error(f"   ❌ File does not exist: {file_path}")
            return None
            
        # Check file size
        file_size = file_path.stat().st_size
        logger.info(f"   File size: {file_size} bytes")
        if file_size < 100:  # Suspiciously small
            logger.warning(f"   ⚠️  File is very small ({file_size} bytes) - might be corrupted")
        
        # Try to read raw content (first few bytes)
        with open(file_path, 'rb') as f:
            raw_content = f.read(100)  # Read first 100 bytes
            logger.info(f"   Raw content (first 100 bytes): {raw_content[:50]}...")
            
            # Check if it's a simple text file with just a number
            try:
                text_content = raw_content.decode('utf-8').strip()
                # Try to parse as number
                float(text_content)
                logger.error(f"   ❌ File appears to contain only a number: {text_content}")
                logger.error(f"   This is NOT a valid model file - it's just a scalar value!")
                return float(text_content)  # Return the scalar to indicate corruption
            except (UnicodeDecodeError, ValueError):
                # Not a simple number, continue with normal loading
                pass
        
        # Try to load with joblib
        loaded_obj = joblib.load(file_path)
        logger.info(f"   Successfully loaded with joblib")
        logger.info(f"   Type of loaded object: {type(loaded_obj)}")
        
        if isinstance(loaded_obj, dict):
            logger.info(f"   Dictionary keys: {list(loaded_obj.keys())}")
            for key, value in loaded_obj.items():
                logger.info(f"     {key}: {type(value)}")
                if hasattr(value, 'shape'):
                    logger.info(f"       shape: {value.shape}")
        elif hasattr(loaded_obj, 'predict'):
            logger.info(f"   Object has predict method - looks like a model")
        else:
            logger.info(f"   Loaded object attributes: {dir(loaded_obj)[:10]}...")  # First 10 attrs
            
        return loaded_obj
        
    except Exception as e:
        logger.error(f"   ❌ Error diagnosing {file_path}: {e}")
        return None

# =====================================================================

# Create error log file
error_log_file = 'error_log.txt'
with open(error_log_file, 'w') as f:
    f.write(f"Error log started at {pd.Timestamp.now()}\n")
    f.write("="*50 + "\n")

def log_error(message):
    with open(error_log_file, 'a') as f:
        f.write(f"{pd.Timestamp.now()}: {message}\n")
    logger.error(message)

class ProteinSelect(Select):
    def accept_residue(self, residue):
        return is_aa(residue, standard=True)

def clean_pdb(input_file, output_file='pure_protein.pdb'):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', input_file)
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_file, ProteinSelect())
        logger.info(f"✅ Clean protein structure saved to {output_file}")
        return output_file
    except Exception as e:
        log_error(f"❌ Failed to clean PDB file {input_file}: {e}")
        return None

def find_all_subdirectories(base_dir):
    subdirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            if full_path.count(os.sep) - base_dir.count(os.sep) == 1:
                subdirs.append(full_path)
    logger.info(f"✅ Found {len(subdirs)} subdirectories in {base_dir}")
    return subdirs

def find_protein_files(directory):
    protein_files = glob.glob(os.path.join(directory, "*_only_protein.pdb"))
    if protein_files:
        logger.info(f"   Found {len(protein_files)} existing only_protein files in {os.path.basename(directory)}")
        return protein_files

    raw_protein_files = glob.glob(os.path.join(directory, "*_protein_protonated.pdb"))
    generated_files = []
    for raw_protein in raw_protein_files:
        if "only" in os.path.basename(raw_protein).lower():
            continue
        base_name = os.path.basename(raw_protein).replace('_protein_protonated.pdb', '')
        only_protein_file = os.path.join(directory, f"{base_name}_only_protein.pdb")
        logger.info(f"   Generating {os.path.basename(only_protein_file)} from {os.path.basename(raw_protein)}")
        result = clean_pdb(raw_protein, only_protein_file)
        if result:
            generated_files.append(result)
    return generated_files

def find_docked_sdf_files(directory):
    # Look for exhaust50_dock.sdf specifically
    sdf_files = glob.glob(os.path.join(directory, "exhaust50_dock.sdf"))
    if sdf_files:
        logger.info(f"   Found {len(sdf_files)} exhaust50 dock SDF files in {os.path.basename(directory)}")
    return sdf_files

def create_single_complex(args):
    protein_file, docked_sdf, output_dir, mol_idx = args
    try:
        supplier = Chem.SDMolSupplier(docked_sdf)
        if len(supplier) <= mol_idx or supplier[mol_idx] is None:
            return None
        
        mol = supplier[mol_idx]
        protein_base = os.path.basename(protein_file).replace('_only_protein.pdb', '')
        docked_base = os.path.basename(docked_sdf).replace('_dock.sdf', '')
        output_file = os.path.join(output_dir, f"{protein_base}_{docked_base}_complex_{mol_idx+1}.pdb")
        
        with open(protein_file, 'r') as protein, open(output_file, 'w') as output:
            for line in protein:
                if line.startswith(('ATOM', 'HETATM')):
                    output.write(line)
            pdb_block = Chem.MolToPDBBlock(mol)
            output.write(pdb_block)
            output.write("END\n")
        return output_file
    except Exception as e:
        log_error(f"❌ Error creating complex {mol_idx+1} from {docked_sdf} with {protein_file}: {e}")
        return None

def create_complexes_for_pair(protein_file, docked_sdf, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    protein_base = os.path.basename(protein_file).replace('_only_protein.pdb', '')
    docked_base = os.path.basename(docked_sdf).replace('_dock.sdf', '')
    
    # First, try to load the SDF file to check if it's valid
    try:
        supplier = Chem.SDMolSupplier(docked_sdf)
        if supplier is None or len(supplier) == 0:
            log_error(f"❌ Invalid or empty SDF file: {docked_sdf}")
            return []
    except Exception as e:
        log_error(f"❌ Error loading SDF file {docked_sdf}: {e}")
        return []
    
    molecules = list(supplier)
    successful_complexes = []
    
    # Prepare arguments for parallel processing
    args_list = [(protein_file, docked_sdf, output_dir, i) for i in range(len(molecules))]
    
    # Process sequentially to avoid nested multiprocessing issues
    for args in tqdm(args_list, desc=f"Creating complexes {protein_base}_{docked_base}", unit="complex", leave=False):
        result = create_single_complex(args)
        if result:
            successful_complexes.append(result)
    
    return successful_complexes

def process_directory_pair(args):
    protein_file, docked_sdf, directory = args
    protein_base = os.path.basename(protein_file).replace('_only_protein.pdb', '')
    docked_base = os.path.basename(docked_sdf).replace('_dock.sdf', '')
    output_dir = os.path.join(directory, f"complexes_{protein_base}_{docked_base}")
    
    logger.info(f"🔗 Creating complexes for {protein_base} + {docked_base}")
    complexes = create_complexes_for_pair(protein_file, docked_sdf, output_dir)
    logger.info(f"   Created {len(complexes)} complexes in {os.path.basename(output_dir)}")
    return output_dir if complexes else None

def process_directory(directory):
    logger.info(f"\n🔍 Processing directory: {directory}")
    logger.info("-" * 50)
    protein_files = find_protein_files(directory)
    docked_sdf_files = find_docked_sdf_files(directory)
    
    if not protein_files or not docked_sdf_files:
        logger.info(f"   Skipping directory {os.path.basename(directory)}: missing protein or dock files")
        return []
    
    # Prepare arguments for parallel processing
    args_list = [(protein_file, docked_sdf, directory) 
                 for protein_file in protein_files 
                 for docked_sdf in docked_sdf_files]
    
    # Process pairs sequentially to avoid nested multiprocessing issues
    complex_dirs = []
    for args in tqdm(args_list, desc=f"Processing {os.path.basename(directory)}", unit="pair"):
        result = process_directory_pair(args)
        if result:
            complex_dirs.append(result)

    return complex_dirs

# -------------------- Dihedral Angle for ARG --------------------
def compute_dihedral_angle(p1, p2, p3, n2):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    n2 = np.array(n2, dtype=float)

    v1 = p2 - p1
    v2 = p3 - p1
    n1 = np.cross(v1, v2)
    norm_n1 = np.linalg.norm(n1)
    if norm_n1 < 1e-8:
        raise ValueError("Charged atoms collinear")
    n1 = n1 / norm_n1

    norm_n2 = np.linalg.norm(n2)
    if norm_n2 < 1e-8:
        raise ValueError("Ring normal zero")
    n2 = n2 / norm_n2

    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(np.abs(cos_angle))
    return np.degrees(angle_rad)

def calculate_angle(ring_normal, charge_vector):
    dot = sum(float(a)*float(b) for a,b in zip(ring_normal, charge_vector))
    norm_r = math.sqrt(sum(float(x)**2 for x in ring_normal))
    norm_c = math.sqrt(sum(float(x)**2 for x in charge_vector))
    if norm_r == 0 or norm_c == 0:
        return 0.0, 0.0
    cos_theta = dot / (norm_r * norm_c)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    angle_deg = math.degrees(math.acos(cos_theta))
    adj = 180 - angle_deg if angle_deg > 90 else angle_deg
    return angle_deg, adj

def calculate_rz(distance, offset):
    try:
        return math.sqrt(max(float(distance)**2 - float(offset)**2, 0))
    except:
        return float('nan')

def analyze_pication_interactions(pdb_file_path):
    from io import StringIO
    import sys
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        pdb_file = Path(pdb_file_path)
        my_mol = PDBComplex()
        my_mol.load_pdb(str(pdb_file))
        my_mol.analyze()
        results = []

        for bs_id, interactions in getattr(my_mol, 'interaction_sets', {}).items():
            pications = getattr(interactions, 'all_pication_laro', [])
            for pication in pications:
                protein_res = pication.restype.strip().upper()
                is_arg = (protein_res == 'ARG')

                ring_normal = np.array(pication.ring.normal, dtype=np.float64)
                charge_center = np.array(pication.charge.center, dtype=np.float64)
                ring_center = np.array(pication.ring.center, dtype=np.float64)
                charge_vector = charge_center - ring_center

                orig_angle, adj_angle = calculate_angle(ring_normal, charge_vector)
                rz = calculate_rz(pication.distance, pication.offset)

                dihedral = float('nan')
                if is_arg:
                    coords = [atom.coords for atom in pication.charge.atoms]
                    if len(coords) >= 3:
                        try:
                            dihedral = compute_dihedral_angle(coords[0], coords[1], coords[2], ring_normal)
                        except:
                            dihedral = float('nan')

                results.append({
                    'Directory': pdb_file.parent.name,
                    'PDB_File': pdb_file.name,
                    'Binding_Site': bs_id,
                    'Ligand': f"{pication.restype_l}-{pication.resnr_l}-{pication.reschain_l}",
                    'Protein': f"{pication.restype}-{pication.resnr}-{pication.reschain}",
                    'Protein_Residue_Type': protein_res,
                    'Is_ARG': is_arg,
                    'Distance': round(float(pication.distance), 2),
                    'Offset': round(float(pication.offset), 2),
                    'RZ': round(rz, 2) if not math.isnan(rz) else float('nan'),
                    'Angle': round(orig_angle, 2),
                    'Adjusted_Angle': round(adj_angle, 2),
                    'Dihedral_Angle': round(dihedral, 2) if not math.isnan(dihedral) else float('nan'),
                    'Ring_Center_X': round(float(ring_center[0]), 3),
                    'Ring_Center_Y': round(float(ring_center[1]), 3),
                    'Ring_Center_Z': round(float(ring_center[2]), 3),
                    'Charged_Center_X': round(float(charge_center[0]), 3),
                    'Charged_Center_Y': round(float(charge_center[1]), 3),
                    'Charged_Center_Z': round(float(charge_center[2]), 3),
                    'Ring_Normal_X': round(float(ring_normal[0]), 3),
                    'Ring_Normal_Y': round(float(ring_normal[1]), 3),
                    'Ring_Normal_Z': round(float(ring_normal[2]), 3),
                    'Ring_Type': pication.ring.type,
                    'Atom_Indices': str(pication.ring.atoms_orig_idx),
                    'Interaction_Type': 'π-Cation'
                })
        sys.stderr = old_stderr
        return results
    except Exception as e:
        sys.stderr = old_stderr
        log_error(f"Error in {pdb_file_path}: {e}")
        return []

def process_single_pdb(pdb_file):
    try:
        return analyze_pication_interactions(pdb_file)
    except Exception as e:
        log_error(f"❌ Error processing {pdb_file}: {e}")
        return []

def process_all_complex_dirs(complex_dirs):
    all_pdb_files = []
    for d in complex_dirs:
        all_pdb_files.extend(glob.glob(os.path.join(d, "*_complex_*.pdb")))
    if not all_pdb_files:
        logger.error("❌ No PDB complexes found!")
        return None

    logger.info(f"🔍 Processing {len(all_pdb_files)} complexes")
    num_proc = min(90, cpu_count())
    all_results = []
    with Pool(processes=num_proc) as pool:
        with tqdm(total=len(all_pdb_files), desc="Analyzing π-cation", unit="file") as pbar:
            for res in pool.imap_unordered(process_single_pdb, all_pdb_files):
                if res:
                    all_results.extend(res)
                pbar.update(1)

    if not all_results:
        logger.info("No π-cation interactions found.")
        return None

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        writer = csv.DictWriter(tmp, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
        logger.info(f"✅ STEP 2 done. Temp file: {tmp.name}")
        return tmp.name

# -------------------- STEP 3: MODEL PREDICTION --------------------
def engineer_arg_features(df):
    df = df.copy()
    df['distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    df['delta_z'] = pd.to_numeric(df['RZ'], errors='coerce')
    df['delta_x'] = pd.to_numeric(df['Offset'], errors='coerce')
    df['dihedral_angle'] = pd.to_numeric(df['Dihedral_Angle'], errors='coerce')
    df['distance'] = df['distance'].clip(lower=1e-3)
    df['inv_distance'] = 1.0 / df['distance']
    df['distance_sq'] = df['distance'] ** 2
    df['delta_z_norm'] = df['delta_z'] / df['distance']
    df['delta_x_norm'] = df['delta_x'] / df['distance']
    dihedral_rad = np.radians(df['dihedral_angle'])
    df['sin_dihedral'] = np.sin(dihedral_rad)
    df['cos_dihedral'] = np.cos(dihedral_rad)
    return df[[
        'delta_z', 'distance', 'inv_distance', 'dihedral_angle',
        'distance_sq', 'delta_x_norm', 'sin_dihedral', 'cos_dihedral', 'delta_z_norm'
    ]]

def run_model_prediction(temp_csv):
    df = pd.read_csv(temp_csv)
    logger.info(f"📊 Loaded {len(df)} interactions")

    # Verify models exist with better error reporting
    if not STANDARD_MODEL_PATH.exists():
        logger.error(f"❌ Standard model not found: {STANDARD_MODEL_PATH}")
        logger.error("Please ensure the trained_models directory is accessible from your current location:")
        logger.error(f"Current working directory: {Path.cwd()}")
        logger.error(f"Expected model location: {STANDARD_MODEL_PATH}")
        sys.exit(1)
    
    if not ARG_MODEL_PATH.exists():
        logger.error(f"❌ ARG model not found: {ARG_MODEL_PATH}")
        logger.error("Please ensure the trained_models directory is accessible from your current location:")
        logger.error(f"Current working directory: {Path.cwd()}")
        logger.error(f"Expected model location: {ARG_MODEL_PATH}")
        sys.exit(1)

    logger.info(f"✅ Using Standard model: {STANDARD_MODEL_PATH}")
    logger.info(f"✅ Using ARG model: {ARG_MODEL_PATH}")

    # Diagnose both model files first
    logger.info("🔍 Diagnosing model files...")
    standard_model_obj = diagnose_model_file(STANDARD_MODEL_PATH)
    arg_model_obj = diagnose_model_file(ARG_MODEL_PATH)

    # Check if diagnosis revealed issues
    if standard_model_obj is not None and isinstance(standard_model_obj, (int, float)):
        logger.error(f"❌ Standard model appears to be corrupted - loaded as scalar value: {standard_model_obj}")
        logger.error("This suggests the .pkl file contains just a number instead of a model object")
    
    if arg_model_obj is not None and isinstance(arg_model_obj, (int, float)):
        logger.error(f"❌ ARG model appears to be corrupted - loaded as scalar value: {arg_model_obj}")
        logger.error("This suggests the .pkl file contains just a number instead of a model object")

    all_preds = []

    # --- Load ARG model ---
    try:
        model_data_arg = joblib.load(ARG_MODEL_PATH)
        if isinstance(model_data_arg, dict):
            model_arg = model_data_arg['model']
            logger.info(f"✅ Loaded ARG model (from dict) from: {ARG_MODEL_PATH}")
        elif hasattr(model_data_arg, 'predict'):
            model_arg = model_data_arg
            logger.info(f"✅ Loaded ARG model (direct) from: {ARG_MODEL_PATH}")
        else:
            logger.error(f"❌ ARG model has unexpected type: {type(model_data_arg)}")
            if isinstance(model_data_arg, (int, float)):
                logger.error("   Model file appears to contain just a number - this is not a valid model file")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to load ARG model: {e}")
        return None

    # --- Load Standard model ---
    try:
        model_data_std = joblib.load(STANDARD_MODEL_PATH)
        if isinstance(model_data_std, dict):
            model_std = model_data_std['model']
            logger.info(f"✅ Loaded Standard model (from dict) from: {STANDARD_MODEL_PATH}")
        elif hasattr(model_data_std, 'predict'):
            model_std = model_data_std
            logger.info(f"✅ Loaded Standard model (direct) from: {STANDARD_MODEL_PATH}")
        else:
            logger.error(f"❌ Standard model has unexpected type: {type(model_data_std)}")
            if isinstance(model_data_std, (int, float)):
                logger.error("   Model file appears to contain just a number - this is not a valid model file")
                logger.error("   Please check if the model file was saved correctly")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to load Standard model: {e}")
        return None

    # --- Process ARG interactions ---
    df_arg = df[df['Is_ARG']].copy()
    if not df_arg.empty:
        X = engineer_arg_features(df_arg)
        valid = X.notnull().all(axis=1)
        if valid.any():
            try:
                pred = model_arg.predict(X[valid])
                df_arg.loc[valid, 'Predicted_Energy'] = pred
                all_preds.append(df_arg[valid])
                logger.info(f"✅ ARG: {valid.sum()} predictions")
            except Exception as e:
                logger.error(f"❌ Error predicting ARG interactions: {e}")

    # --- Process Non-ARG interactions ---
    df_other = df[~df['Is_ARG']].copy()
    if not df_other.empty:
        cols = ['RZ', 'Offset', 'Adjusted_Angle', 'Distance']
        X = df_other[cols].copy()
        X.columns = ['delta_z', 'delta_x', 'dihedral_angle', 'distance']
        valid = X.notnull().all(axis=1)
        if valid.any():
            try:
                X_valid = X[valid][['delta_z', 'delta_x', 'dihedral_angle', 'distance']]
                pred = model_std.predict(X_valid)
                df_other.loc[valid, 'Predicted_Energy'] = pred
                all_preds.append(df_other[valid])
                logger.info(f"✅ Non-ARG: {valid.sum()} predictions")
            except Exception as e:
                logger.error(f"❌ Error predicting Non-ARG interactions: {e}")

    if not all_preds:
        logger.error("❌ No valid predictions!")
        return None

    df_final = pd.concat(all_preds, ignore_index=True)
    df_final['PDB_ID'] = df_final['PDB_File'].str.extract(r'([^_]+_[^_]+)_')[0]
    df_final['Energy_Rank'] = df_final.groupby('PDB_ID')['Predicted_Energy'].rank(method='dense', ascending=True).astype(int)
    df_final = df_final.sort_values(['PDB_ID', 'Energy_Rank']).reset_index(drop=True)

    # Round numeric columns
    num_cols = ['Distance','Offset','RZ','Angle','Adjusted_Angle','Dihedral_Angle','Predicted_Energy',
                'Ring_Center_X','Ring_Center_Y','Ring_Center_Z',
                'Charged_Center_X','Charged_Center_Y','Charged_Center_Z',
                'Ring_Normal_X','Ring_Normal_Y','Ring_Normal_Z']
    for c in num_cols:
        if c in df_final.columns:
            df_final[c] = pd.to_numeric(df_final[c], errors='coerce').round(2)

    output_file = 'all_sampled_poses_with-pi-cation-interactions.csv'
    df_final.to_csv(output_file, index=False)
    logger.info(f"💾 Final results saved to: {output_file}")
    return output_file

def cleanup_generated_files_recursive(base_dir):
    patterns = ['plipfixed.*.pdb', '*_complex_*_protonated.pdb']
    count = 0
    for pattern in patterns:
        for f in glob.glob(os.path.join(base_dir, '**', pattern), recursive=True):
            try:
                os.remove(f); count += 1
            except: pass
        for f in glob.glob(os.path.join(os.path.dirname(base_dir), pattern)):
            try:
                os.remove(f); count += 1
            except: pass
    if count: logger.info(f"🧹 Cleaned {count} temp files")


# ==================== MAIN ====================
if __name__ == "__main__":

    base_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    logger.info(f"📁 Base directory: {base_dir}")

    # Log current directory structure for debugging
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"Parent directory: {Path.cwd().parent}")
    
    # The new get_model_paths() function will handle the search automatically
    # But let's show what it found for transparency
    logger.info(f"Model paths determined by robust search:")
    logger.info(f"  Standard model: {STANDARD_MODEL_PATH}")
    logger.info(f"  ARG model: {ARG_MODEL_PATH}")
    
    # Check if the resolved paths actually exist
    if STANDARD_MODEL_PATH.exists():
        logger.info(f"✅ Standard model file confirmed to exist")
    else:
        logger.warning(f"⚠️  Standard model file not found at resolved path")
    
    if ARG_MODEL_PATH.exists():
        logger.info(f"✅ ARG model file confirmed to exist")
    else:
        logger.warning(f"⚠️  ARG model file not found at resolved path")

    # STEP 1
    subdirs = find_all_subdirectories(base_dir)
    if not subdirs:
        logger.error("❌ No subdirectories found!")
        sys.exit(1)

    print("  STEP 1: GENERATING COMPLEXES")
    print("-" * 50)
    all_complex_dirs = []
    for d in tqdm(subdirs, desc="Directories", unit="dir"):
        try:
            complex_dirs = process_directory(d)
            all_complex_dirs.extend([cd for cd in complex_dirs if cd is not None])
        except Exception as e:
            log_error(f"❌ Error processing directory {d}: {e}")
            continue

    if not all_complex_dirs:
        logger.error("❌ No complexes generated!")
        sys.exit(1)

    # STEP 2
    print("  STEP 2: ANALYZING π-CATION INTERACTIONS")
    print("-" * 50)
    temp_csv = process_all_complex_dirs(all_complex_dirs)
    if not temp_csv:
        logger.info("No interactions → exiting.")
        sys.exit(0)

    # STEP 3
    print("  STEP 3: MODEL PREDICTION AND RANKING")
    print("-" * 50)
    final_file = run_model_prediction(temp_csv)
    if final_file:
        logger.info("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        logger.info(f"Error log saved to: {error_log_file}")
    else:
        logger.error("💥 STEP 3 FAILED!")
        logger.info(f"Error log saved to: {error_log_file}")
        sys.exit(1)

    # Cleanup
    try:
        os.unlink(temp_csv)
        logger.info(f"🗑️  Removed temp file: {temp_csv}")
    except:
        pass
    cleanup_generated_files_recursive(base_dir)
