# Baseline V11: Post-Proc Integration Guide for Trainer

## Overview

**Stage:** Phase 2C (post-processing variant implementation & full LOEO validation)  
**Trainer Task:** Implement exp#1-3 post-proc transformations on pilkwang frozen predictions  
**Input:** Full LOEO GEFFS (199 datasets) from Phase 2B  
**Output:** Variant-specific predictions + official_scorer + metric_anatomy metrics  
**Timeline:** ~2-3h (GPU inference already running in Phase 2B)

---

## Configuration Structure

### Location
```
baseline/experiments_v11/
  ├── exp1_edge_threshold_gap_recovery.yaml
  ├── exp2_centroid_refine_smooth.yaml
  └── exp3_division_filter.yaml
```

### Format (YAML)
Each config specifies:
- **name**: Experiment identifier (stage2_expX_*)
- **description**: Human-readable summary of transformations
- **input**: Path to pilkwang predictions (GEFFS)
- **postproc**: Post-processing parameters (transformation-specific)
- **output**: Output directory for variant predictions
- **evaluation**: Scorer config (split file, fold, enable anatomy)
- **tracking**: MLflow logging config

### Example: exp1_edge_threshold_gap_recovery.yaml
```yaml
name: stage2_exp1_edge_threshold_gap_recovery
input:
  predictions_dir: /path/to/pilkwang/split_0
postproc:
  edge_length_max_um: 12.0
  gap_recovery_enabled: true
  gap_recovery_max_distance_um: 6.0
output:
  predictions_dir: /path/to/exp1_predictions
evaluation:
  use_official_scorer: true
  use_metric_anatomy: true
```

---

## Post-Proc Code Skeleton

### Module: `baseline/postproc.py`

```python
"""Post-processing transformations for stage2 variants.

Entry point: apply_variant(variant_name, config, predictions_dir, gt_dir)
"""

import json
from pathlib import Path
import numpy as np
from src import io, metric


class PostProcTransform:
    """Base class for post-processing transformations."""

    def __init__(self, config: dict):
        self.config = config

    def apply(self, nodes: np.ndarray, edges: np.ndarray) -> tuple:
        """Apply transformation to nodes + edges.
        
        Parameters
        ----------
        nodes : np.ndarray
            (N, 5) array [node_id, t, z, y, x]
        edges : np.ndarray
            (E, 2) array [source_id, target_id]

        Returns
        -------
        nodes_out, edges_out
            Transformed arrays
        """
        raise NotImplementedError

    def __call__(self, nodes, edges):
        return self.apply(nodes, edges)


class EdgeThresholdGapRecovery(PostProcTransform):
    """exp#1: Edge-length threshold + gap-recovery.
    
    Filters edges by physical distance. Re-links broken tracks in local neighborhood.
    """

    def apply(self, nodes, edges):
        # Step 1: Filter edges by length
        edges_filtered = self._filter_by_length(nodes, edges)
        
        # Step 2: Gap recovery (re-link broken tracks)
        edges_recovered = self._gap_recovery(nodes, edges_filtered)
        
        return nodes, edges_recovered

    def _filter_by_length(self, nodes, edges):
        """Remove edges > edge_length_max_um."""
        max_um = self.config.get("edge_length_max_um", 12.0)
        # Compute distances, filter
        # TODO: implement
        return edges

    def _gap_recovery(self, nodes, edges):
        """Re-link tracks with gaps < max_distance."""
        # TODO: implement motion-aware re-linking
        return edges


class CentroidRefineSmooth(PostProcTransform):
    """exp#2: Centroid refinement + motion smoothing.
    
    Refines node positions via cluster statistics. Smooths trajectories via motion model.
    """

    def apply(self, nodes, edges):
        # Step 1: Refine centroids
        nodes_refined = self._refine_centroids(nodes)
        
        # Step 2: Motion smoothing
        nodes_smooth = self._motion_smooth(nodes_refined, edges)
        
        return nodes_smooth, edges

    def _refine_centroids(self, nodes):
        """Sub-voxel centroid refinement."""
        # TODO: implement cluster statistics
        return nodes

    def _motion_smooth(self, nodes, edges):
        """Smooth trajectories via velocity consistency."""
        # TODO: implement motion model
        return nodes


class DivisionFilter(PostProcTransform):
    """exp#3: Division-aware edge filtering.
    
    Identifies divisions, filters geometrically impossible edges, suppresses FP near divisions.
    """

    def apply(self, nodes, edges):
        # Step 1: Identify divisions
        divisions = self._identify_divisions(nodes, edges)
        
        # Step 2: Filter edges violating division geometry
        edges_filtered = self._filter_by_division_geometry(nodes, edges, divisions)
        
        # Step 3: Suppress FP edges near divisions
        edges_final = self._suppress_fp_near_divisions(nodes, edges_filtered, divisions)
        
        return nodes, edges_final

    def _identify_divisions(self, nodes, edges):
        """Find parent → 2 daughter events."""
        # TODO: implement division detection
        return []

    def _filter_by_division_geometry(self, nodes, edges, divisions):
        """Remove edges that violate division constraints."""
        # TODO: implement geometry filtering
        return edges

    def _suppress_fp_near_divisions(self, nodes, edges, divisions):
        """Reduce FP edges in division neighborhoods."""
        # TODO: implement FP suppression
        return edges


def apply_variant(variant_name, config, predictions_dir, output_dir):
    """Apply post-proc variant to all predictions.
    
    Parameters
    ----------
    variant_name : str
        One of: 'exp1_edge_threshold_gap_recovery', 'exp2_centroid_refine_smooth', 'exp3_division_filter'
    config : dict
        Variant config (from YAML)
    predictions_dir : Path
        Input pilkwang predictions directory
    output_dir : Path
        Output directory for variant predictions
    
    Returns
    -------
    dict
        Metrics dict with official_score, anatomy results
    """
    
    # Map variant names to classes
    variants = {
        "exp1_edge_threshold_gap_recovery": EdgeThresholdGapRecovery,
        "exp2_centroid_refine_smooth": CentroidRefineSmooth,
        "exp3_division_filter": DivisionFilter,
    }
    
    if variant_name not in variants:
        raise ValueError(f"Unknown variant: {variant_name}")
    
    transform_class = variants[variant_name]
    transform = transform_class(config.get("postproc", {}))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply transformation to all predictions
    results = {}
    for pred_geff in sorted(predictions_dir.glob("*.geff")):
        # Load prediction
        nodes, edges = io.read_geff(pred_geff)
        
        # Apply transformation
        nodes_t, edges_t = transform(nodes, edges)
        
        # Save transformed prediction
        output_geff = output_dir / pred_geff.name
        io.write_geff(output_geff, nodes_t, edges_t)
        
        results[pred_geff.name] = "success"
    
    return results
```

---

## Trainer Integration Workflow

### Step 1: Load Configuration
```python
import yaml
from pathlib import Path

config_path = Path("baseline/experiments_v11/exp1_edge_threshold_gap_recovery.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)
```

### Step 2: Apply Variant
```python
from baseline.postproc import apply_variant

variant_name = "exp1_edge_threshold_gap_recovery"
predictions_dir = Path("output/stage2/loeo_predictions")
output_dir = Path("output/stage2/exp1_predictions")

results = apply_variant(variant_name, config, predictions_dir, output_dir)
```

### Step 3: Score Variant
```python
from fleet_agents.official_scorer import score_datasets
from fleet_agents.metric_anatomy import anatomy
import json

# Load LOEO split
with open("learning/ensemble_work/finetune/fleet_loeo_mini.json") as f:
    folds = json.load(f)

# Score both folds
for fold_idx in range(2):
    test_datasets = folds[fold_idx]["test"]
    
    agg_score, score_rows = score_datasets(test_datasets, output_dir)
    agg_anatomy, anatomy_rows = anatomy(test_datasets, output_dir)
    
    print(f"Fold {fold_idx}: {agg_score['score']:.4f}")
    print(f"  R_node={agg_anatomy['R_node']:.4f}, R_edge={agg_anatomy['R_edge']:.4f}")
```

### Step 4: Log Results
```python
import json

results_json = {
    "variant": variant_name,
    "fold0_score": fold0_score,
    "fold1_score": fold1_score,
    "avg_score": (fold0_score + fold1_score) / 2,
    "delta_from_baseline": avg_score - 0.8527,
}

with open(f"output/stage2/{variant_name}_results.json", "w") as f:
    json.dump(results_json, f, indent=2)
```

---

## Runner Script Template

### `baseline/run_experiments_v11.sh`

```bash
#!/bin/bash
# Run all Stage 2 post-proc variants (exp#1-3) on full LOEO

set -e

export PYTHONPATH="tools/researchpapers:$PYTHONPATH"
export PYTHONPATH="learning/ensemble_work:$PYTHONPATH"

COMP_ROOT=$(pwd)
OUTPUT_DIR="$COMP_ROOT/output/stage2"

# Ensure Phase 2B predictions exist
if [ ! -d "$OUTPUT_DIR/loeo_predictions" ]; then
    echo "ERROR: Phase 2B predictions not found at $OUTPUT_DIR/loeo_predictions"
    exit 1
fi

# Run each variant
for exp in exp1_edge_threshold_gap_recovery exp2_centroid_refine_smooth exp3_division_filter; do
    echo "Running $exp..."
    
    python baseline/run_baseline.py \
        --config "baseline/experiments_v11/${exp}.yaml" \
        --fold 0 \
        --fold 1 \
        --score \
        --output-dir "$OUTPUT_DIR/${exp}"
done

echo "All variants complete. Results in $OUTPUT_DIR/"
```

---

## File Paths Summary

| Component | Path |
|-----------|------|
| Configs | `baseline/experiments_v11/exp{1,2,3}_*.yaml` |
| Post-proc code | `baseline/postproc.py` |
| Input (Phase 2B) | `output/stage2/loeo_predictions/` (199 .geff files) |
| Output (exp#1) | `output/stage2/exp1_predictions/` |
| Output (exp#2) | `output/stage2/exp2_predictions/` |
| Output (exp#3) | `output/stage2/exp3_predictions/` |
| Results JSON | `output/stage2/{exp}_results.json` |
| Runner | `baseline/run_experiments_v11.sh` |

---

## Integration Points

### Data Flow
```
pilkwang GEFFS (phase2b) 
  → apply_variant(exp#i) 
  → variant predictions 
  → official_scorer + metric_anatomy 
  → results.json + MLflow
```

### Configuration Parametrization
- **Edge threshold:** `config["postproc"]["edge_length_max_um"]`
- **Gap recovery:** `config["postproc"]["gap_recovery_*"]`
- **Centroid refinement:** `config["postproc"]["centroid_refinement_*"]`
- **Motion smoothing:** `config["postproc"]["motion_smoothing_*"]`
- **Division filtering:** `config["postproc"]["division_*"]`

### Scoring Integration
- Use `official_scorer.score_datasets()` on variant output
- Use `metric_anatomy.anatomy()` for decomposition
- Log to MLflow (config: `tracking.mlflow_enabled`)

---

## Next Steps (Trainer)

1. Implement `baseline/postproc.py` with exp#1-3 logic
2. Update `baseline/run_experiments_v11.sh` with trainer's submission workflow
3. Await Phase 2B completion signal (output directory populated with 199 .geff files)
4. Run exp#1-3 in parallel or sequence on full LOEO
5. Report winning variant + delta to leader

---

*Generated: Phase 2C prep documentation for trainer integration.*
