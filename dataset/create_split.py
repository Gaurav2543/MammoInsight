import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import logging
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CSV = "mammo-bench.csv"
OUTPUT_CSV = "mammo-bench_master_split.csv"
SEED = 42

class StatsTracker:
    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(int))
    def log(self, dataset, reason):
        self.stats[dataset][reason] += 1
        self.stats['TOTAL'][reason] += 1
    def print_report(self):
        print("\n" + "="*80)
        print("DATA EXCLUSION REPORT")
        print("="*80)
        print(f"{'Dataset':<20} | {'Total':<10} | {'Kept':<10} | {'Miss PID':<10} | {'Miss View':<10} | {'Miss Label':<10}")
        print("-" * 80)
        for dset in sorted(self.stats.keys()):
            if dset == 'TOTAL': continue
            s = self.stats[dset]
            print(f"{dset:<20} | {s['total_rows']:<10} | {s['kept_rows']:<10} | {s['missing_pid']:<10} | {s['missing_view']:<10} | {s['missing_label']:<10}")
        print("-" * 80)
        t = self.stats['TOTAL']
        print(f"{'TOTAL':<20} | {t['total_rows']:<10} | {t['kept_rows']:<10} | {t['missing_pid']:<10} | {t['missing_view']:<10} | {t['missing_label']:<10}")
        print("="*80 + "\n")

def get_patient_identifier(row, source_dataset, row_idx):
    """
    Robust ID extraction for all datasets.
    """
    # 0. DMID (Unique per image)
    if source_dataset == 'dmid':
        return f"dmid_{row_idx}"

    # 1. Try standard columns first
    if pd.notna(row.get('study_id')): return str(row.get('study_id'))
    if pd.notna(row.get('patientID')): return str(row.get('patientID'))
    if pd.notna(row.get('patient_id')): return str(row.get('patient_id'))
    if pd.notna(row.get('source_subjectID')): return str(row.get('source_subjectID'))

    # 2. INbreast fallback
    if source_dataset == 'inbreast':
        path = str(row.get('original_source_path', ''))
        try:
            return path.split('/')[-1].split('_')[1]
        except:
            pass
            
    # 3. Last resort for VinDr/RSNA if columns are messy (use index to keep data)
    if source_dataset in ['vindr-mammo', 'rsna-screening']:
        return f"{source_dataset}_{row_idx}"

    return None

def create_master_split():
    tracker = StatsTracker()
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    breast_groups = {}
    # Track all breast_ids for each base_id to handle duplicates
    base_to_breast_ids = defaultdict(list)
    
    print("Grouping images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        dset = str(row.get('source_dataset', 'unknown'))
        tracker.stats[dset]['total_rows'] += 1
        tracker.stats['TOTAL']['total_rows'] += 1

        # 1. ID Check
        pid = get_patient_identifier(row, dset, idx)
        if pid is None:
            tracker.log(dset, 'missing_pid')
            continue

        # 2. View Check
        view = row.get('view')
        if pd.isna(view):
            tracker.log(dset, 'missing_view')
            continue
            
        # 3. Label Check - RELAXED LOGIC
        classification = row.get('classification')
        roi_path = row.get('ROI_path')
        
        if pd.isna(classification):
            if pd.notna(roi_path):
                classification = 'Unknown'
            elif dset == 'rsna-screening' and pd.notna(row.get('cancer')):
                classification = 'Malignant' if int(row['cancer']) == 1 else 'Normal'
            else:
                if pd.isna(row.get('density')) and pd.isna(row.get('BIRADS')):
                    tracker.log(dset, 'missing_label')
                    continue
                classification = 'Unknown'

        tracker.log(dset, 'kept_rows')
        
        # Normalize view
        view_normalized = str(view).upper().strip()
        
        # Determine slot
        if view_normalized in ['CC']:
            slot = 'cc'
        elif view_normalized in ['MLO']:
            slot = 'mlo'
        else:
            raise ValueError(f"Unknown view '{view}' in dataset '{dset}' at index {idx}")
        
        # Base ID
        lat = row.get('laterality', 'Unknown')
        base_id = f"{dset}_{pid}_{lat}"
        
        # Find an existing breast_id with an empty slot, or create new one
        breast_id = None
        for existing_id in base_to_breast_ids[base_id]:
            if slot == 'cc' and breast_groups[existing_id]['cc_path'] is None:
                breast_id = existing_id
                break
            elif slot == 'mlo' and breast_groups[existing_id]['mlo_path'] is None:
                breast_id = existing_id
                break
        
        # If no available slot found, create new breast_id
        if breast_id is None:
            if len(base_to_breast_ids[base_id]) == 0:
                breast_id = base_id
            else:
                breast_id = f"{base_id}_dup{len(base_to_breast_ids[base_id])}"
            
            base_to_breast_ids[base_id].append(breast_id)
            
            # Create new breast entry
            breast_groups[breast_id] = {
                'breast_id': breast_id,
                'patient_id': pid,
                'source_dataset': dset,
                'laterality': lat,
                'age': row.get('subject_age', 0),
                'density': row.get('density'),
                'classification': classification,
                'birads': row.get('BIRADS'),
                'abnormality': row.get('abnormality'),
                'molecular_subtype': row.get('molecular_subtype'),
                'cc_path': None, 'mlo_path': None,
                'cc_roi': None, 'mlo_roi': None
            }
        
        # Assign to slot
        if slot == 'cc':
            breast_groups[breast_id]['cc_path'] = row['preprocessed_image_path']
            breast_groups[breast_id]['cc_roi'] = row.get('ROI_path')
        else:
            breast_groups[breast_id]['mlo_path'] = row['preprocessed_image_path']
            breast_groups[breast_id]['mlo_roi'] = row.get('ROI_path')

    tracker.print_report()
    
    # Convert to DataFrame
    breast_df = pd.DataFrame.from_dict(breast_groups, orient='index')
    
    # ── Filter invalid label values ──────────────────────────────────────
    pre_filter = len(breast_df)
    
    # Remove 'Suspicious Malignant' and 'Unknown' classification
    breast_df = breast_df[~breast_df['classification'].isin(['Suspicious Malignant'])]
    
    # Remove abnormality values not in {normal, mass, calcification, NaN}
    valid_abnorm = ['normal', 'mass', 'calcification']
    breast_df = breast_df[breast_df['abnormality'].isna() | breast_df['abnormality'].isin(valid_abnorm)]
    
    # Remove invalid density values (keep A/B/C/D and NaN)
    valid_density = ['A', 'B', 'C', 'D']
    breast_df = breast_df[breast_df['density'].isna() | breast_df['density'].isin(valid_density)]
    
    # Remove BI-RADS > 5 (keep 0-5 and NaN)
    def _valid_birads(val):
        if pd.isna(val): return True
        try: return 0 <= int(float(val)) <= 5
        except: return False
    breast_df = breast_df[breast_df['birads'].apply(_valid_birads)]
    
    breast_df = breast_df.reset_index(drop=True)
    # ── Detailed label statistics ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MASTER SPLIT LABEL STATISTICS (breast-level)")
    print(f"{'='*60}")
    print(f"Total breast cases: {len(breast_df)}")
    
    print(f"\nClassification:")
    print(breast_df['classification'].value_counts(dropna=False).to_string())
    
    print(f"\nDensity:")
    print(breast_df['density'].value_counts(dropna=False).to_string())
    
    print(f"\nAbnormality:")
    print(breast_df['abnormality'].value_counts(dropna=False).to_string())
    
    print(f"\nBI-RADS:")
    print(breast_df['birads'].value_counts(dropna=False).to_string())
    
    print(f"\nMolecular:")
    print(breast_df['molecular_subtype'].value_counts(dropna=False).to_string())
    
    # Show how many cases have EACH label available
    print(f"\n{'='*60}")
    print("LABEL AVAILABILITY (non-NaN counts)")
    print(f"{'='*60}")
    for col in ['classification', 'density', 'birads', 'abnormality', 'molecular_subtype']:
        valid = breast_df[col].notna().sum()
        mappable = 0
        if col == 'classification':
            mappable = breast_df[col].isin(['Normal', 'Benign', 'Malignant']).sum()
        elif col == 'density':
            mappable = breast_df[col].isin(['A', 'B', 'C', 'D']).sum()
        elif col == 'abnormality':
            mappable = breast_df[col].isin(['normal', 'mass', 'calcification']).sum()
        elif col == 'birads':
            def _ok(v):
                try: return 0 <= int(float(v)) <= 5
                except: return False
            mappable = breast_df[col].dropna().apply(_ok).sum()
        elif col == 'molecular_subtype':
            mappable = breast_df[col].isin(['Luminal A', 'Luminal B', 'HER2-enriched', 'triple negative']).sum()
        print(f"  {col:<22}: {valid:>6} non-NaN, {mappable:>6} mappable (validity=1)")
    print(f"{'='*60}\n")
    
    print(f"\nLabel filtering: {pre_filter} → {len(breast_df)} (removed {pre_filter - len(breast_df)})")
    
    # Count unpaired cases
    cc_only = breast_df[(breast_df['cc_path'].notna()) & (breast_df['mlo_path'].isna())]
    mlo_only = breast_df[(breast_df['cc_path'].isna()) & (breast_df['mlo_path'].notna())]
    both_views = breast_df[(breast_df['cc_path'].notna()) & (breast_df['mlo_path'].notna())]

    print(f"\nView pairing statistics:")
    print(f"Both CC and MLO: {len(both_views)}")
    print(f"CC only: {len(cc_only)}")
    print(f"MLO only: {len(mlo_only)}")
    print(f"Total images: {len(both_views)*2 + len(cc_only) + len(mlo_only)}")
    print(f"Total Unique Breast Cases: {len(breast_df)}")
    
    print("\nPerforming RANDOM splits (75/10/15)...")
    
    # Simple Random Split
    train_df, temp_df = train_test_split(breast_df, test_size=0.25, random_state=SEED, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.6, random_state=SEED, shuffle=True)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    final_df = pd.concat([train_df, val_df, test_df])
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Saved to {OUTPUT_CSV}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    create_master_split()