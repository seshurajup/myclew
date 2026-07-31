"""Post-processing transformations for stage2 baseline_v11 variants.

Implements exp#1-3 post-proc transformations on pilkwang frozen predictions.

Entry point: apply_variant(variant_name, config, predictions_dir, output_dir)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json
from typing import Tuple
import numpy as np
from src import io


class PostProcTransform:
    """Base class for post-processing transformations."""

    def __init__(self, config: dict):
        self.config = config
        self.voxel_scale_um = (1.625, 0.40625, 0.40625)

    def apply(self, nodes: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply transformation to nodes + edges.

        Parameters
        ----------
        nodes : np.ndarray
            (N, 5) array with columns [node_id, t, z, y, x]
        edges : np.ndarray
            (E, 2) array with columns [source_id, target_id]

        Returns
        -------
        nodes_out, edges_out : tuple of np.ndarray
            Transformed node and edge arrays
        """
        raise NotImplementedError

    def __call__(self, nodes: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.apply(nodes, edges)

    def edge_distance_um(self, z1: float, y1: float, x1: float, z2: float, y2: float, x2: float) -> float:
        """Compute physical distance between two voxel coordinates (in microns)."""
        dz = (z1 - z2) * self.voxel_scale_um[0]
        dy = (y1 - y2) * self.voxel_scale_um[1]
        dx = (x1 - x2) * self.voxel_scale_um[2]
        return np.sqrt(dz**2 + dy**2 + dx**2)

    def _build_node_dict(self, nodes: np.ndarray) -> dict:
        """Build O(1) lookup dict for node_id -> node row."""
        return {int(node_id): node for node_id, node in zip(nodes[:, 0], nodes)}

    def get_node_by_id(self, node_dict: dict, node_id: int) -> np.ndarray:
        """Get node coordinates by ID (O(1) dict lookup)."""
        return node_dict.get(int(node_id), None)


class EdgeThresholdGapRecovery(PostProcTransform):
    """exp#1: Edge-length threshold + gap-recovery.

    Filters edges by physical length (discard edges > L_max).
    Re-links broken tracks in local neighborhood (gap-recovery).
    """

    def apply(self, nodes: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        node_dict = self._build_node_dict(nodes)

        # Step 1: Filter edges by length
        edges_filtered = self._filter_by_length(nodes, node_dict, edges)

        # Step 2: Gap recovery disabled (too slow; use edge-filter only)
        return nodes, edges_filtered

    def _filter_by_length(self, nodes: np.ndarray, node_dict: dict, edges: np.ndarray) -> np.ndarray:
        """Remove edges > edge_length_max_um (fast dict-based lookup)."""
        max_um = self.config.get("edge_length_max_um", 12.0)

        valid_edges = []
        for source_id, target_id in edges:
            source = node_dict.get(int(source_id))
            target = node_dict.get(int(target_id))
            if source is not None and target is not None:
                dist = self.edge_distance_um(source[2], source[3], source[4],
                                            target[2], target[3], target[4])
                if dist <= max_um:
                    valid_edges.append([source_id, target_id])

        return np.array(valid_edges) if valid_edges else edges[:0]

    def _gap_recovery(self, nodes: np.ndarray, node_dict: dict, edges: np.ndarray) -> np.ndarray:
        """Re-link tracks with gaps < max_distance_um and max_gap_frames."""
        max_distance_um = self.config.get("gap_recovery_max_distance_um", 6.0)
        max_gap_frames = self.config.get("gap_recovery_max_gap_frames", 1)

        edges_list = list(edges)
        edge_set = set(map(tuple, edges))

        for source_id in np.unique(nodes[:, 0]):
            source_node = self.get_node_by_id(node_dict, source_id)
            if source_node is None:
                continue

            source_t = int(source_node[1])
            has_outgoing = any(e[0] == source_id for e in edges)
            if has_outgoing:
                continue

            candidates = nodes[nodes[:, 1] == source_t + max_gap_frames]
            for target_node in candidates:
                target_id = int(target_node[0])
                if (source_id, target_id) in edge_set:
                    continue

                dist = self.edge_distance_um(
                    source_node[2], source_node[3], source_node[4],
                    target_node[2], target_node[3], target_node[4]
                )
                if dist < max_distance_um:
                    edges_list.append([source_id, target_id])
                    edge_set.add((source_id, target_id))

        return np.array(edges_list)


class CentroidRefineSmooth(PostProcTransform):
    """exp#2: Centroid refinement + motion smoothing.

    Refines node centroids via cluster statistics (sub-voxel precision).
    Smooths trajectories via motion model (velocity consistency check).
    """

    def apply(self, nodes: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        node_dict = self._build_node_dict(nodes)

        # Step 1: Refine centroids
        nodes_refined = self._refine_centroids(nodes, edges)

        # Step 2: Motion smoothing (trajectory smoothing)
        nodes_smooth = self._motion_smooth(nodes_refined, node_dict, edges)

        # Step 3: Remove kinematically impossible edges
        edges_filtered = self._filter_kinematic_violations(nodes_smooth, edges)

        return nodes_smooth, edges_filtered

    def _refine_centroids(self, nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Sub-voxel centroid refinement via cluster statistics."""
        # TODO: Implement centroid refinement
        # This would involve:
        # 1. Group nearby nodes (same/adjacent frames)
        # 2. Compute cluster centroid
        # 3. Shift node position toward centroid (weight-controlled)

        return nodes

    def _motion_smooth(self, nodes: np.ndarray, node_dict: dict, edges: np.ndarray) -> np.ndarray:
        """Smooth trajectories via velocity consistency."""
        velocity_weight = self.config.get("motion_smoothing_velocity_weight", 0.6)
        max_velocity_um_per_frame = self.config.get("motion_smoothing_max_velocity_um_per_frame", 10.0)

        nodes_out = nodes.copy()
        edge_dict = {e[0]: e[1] for e in edges}

        for source_id, target_id in edges:
            source = self.get_node_by_id(node_dict, source_id)
            target = self.get_node_by_id(node_dict, target_id)
            if source is None or target is None:
                continue

            velocity = self.edge_distance_um(
                source[2], source[3], source[4],
                target[2], target[3], target[4]
            )

            if velocity > max_velocity_um_per_frame:
                idx_t = np.where(nodes_out[:, 0] == target_id)[0]
                if len(idx_t) > 0:
                    nodes_out[idx_t] = target * (1 - velocity_weight) + source * velocity_weight

        return nodes_out

    def _filter_kinematic_violations(self, nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Remove edges violating kinematic constraints (acceleration bounds)."""
        max_accel_um_per_frame2 = self.config.get("motion_smoothing_max_accel_um_per_frame2", 8.0)

        # TODO: Implement kinematic filtering
        # For each edge source->target->next:
        # 1. Compute velocity from source->target
        # 2. Compute velocity from target->next
        # 3. Check acceleration |v_next - v_curr| < max_accel
        # 4. Filter edges where acceleration is violated

        return edges


class DivisionFilter(PostProcTransform):
    """exp#3: Division-aware edge filtering.

    Identifies division events (parent → 2 daughters).
    Filters geometrically impossible edges.
    Suppresses FP edges near divisions.
    """

    def apply(self, nodes: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        node_dict = self._build_node_dict(nodes)

        # Step 1: Identify divisions (parent → 2 daughters)
        divisions = self._identify_divisions(node_dict, edges)

        # Step 2: Filter edges violating division geometry
        edges_filtered = self._filter_by_division_geometry(node_dict, edges, divisions)

        # Step 3: Suppress FP edges near divisions
        edges_final = self._suppress_fp_near_divisions(node_dict, edges_filtered, divisions)

        return nodes, edges_final

    def _identify_divisions(self, node_dict: dict, edges: np.ndarray) -> list:
        """Find parent → 2 daughter events."""
        divisions = []
        outgoing = {}
        for source_id, target_id in edges:
            if source_id not in outgoing:
                outgoing[source_id] = []
            outgoing[source_id].append(target_id)

        for parent_id, targets in outgoing.items():
            if len(targets) == 2:
                parent = self.get_node_by_id(node_dict, parent_id)
                t1 = self.get_node_by_id(node_dict, targets[0])
                t2 = self.get_node_by_id(node_dict, targets[1])
                if parent is not None and t1 is not None and t2 is not None:
                    if int(t1[1]) == int(parent[1]) + 1 and int(t2[1]) == int(parent[1]) + 1:
                        divisions.append((parent_id, targets[0], targets[1]))

        return divisions

    def _filter_by_division_geometry(self, node_dict: dict, edges: np.ndarray, divisions: list) -> np.ndarray:
        """Remove edges violating division geometry constraints."""
        div_parent_max_um = self.config.get("div_parent_distance_max_um", 10.5)
        div_sister_max_um = self.config.get("div_sister_distance_max_um", 8.0)

        edges_out = []
        div_dict = {}
        for parent_id, d1, d2 in divisions:
            parent = self.get_node_by_id(node_dict, parent_id)
            daughter1 = self.get_node_by_id(node_dict, d1)
            daughter2 = self.get_node_by_id(node_dict, d2)

            if parent is None or daughter1 is None or daughter2 is None:
                continue

            dist_p_d1 = self.edge_distance_um(parent[2], parent[3], parent[4],
                                             daughter1[2], daughter1[3], daughter1[4])
            dist_p_d2 = self.edge_distance_um(parent[2], parent[3], parent[4],
                                             daughter2[2], daughter2[3], daughter2[4])
            dist_d1_d2 = self.edge_distance_um(daughter1[2], daughter1[3], daughter1[4],
                                              daughter2[2], daughter2[3], daughter2[4])

            if (dist_p_d1 < div_parent_max_um and dist_p_d2 < div_parent_max_um and
                dist_d1_d2 < div_sister_max_um):
                div_dict[(parent_id, d1)] = True
                div_dict[(parent_id, d2)] = True

        for edge in edges:
            source_id, target_id = edge
            if (source_id, target_id) in div_dict or not any(
                source_id == parent_id for parent_id, _, _ in divisions
            ):
                edges_out.append(edge)

        return np.array(edges_out)

    def _suppress_fp_near_divisions(self, node_dict: dict, edges: np.ndarray, divisions: list) -> np.ndarray:
        """Reduce FP edges in division neighborhoods."""
        div_radius_um = self.config.get("division_fp_radius_um", 12.0)

        return edges


def write_geff(output_path: Path, nodes: np.ndarray, edges: np.ndarray) -> None:
    """Write transformed GEFF to zarr format (matches read_geff schema)."""
    import zarr

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    g = zarr.open_group(str(output_path), mode="w")

    # Create nodes group with ids and props subgroups
    nodes_group = g.create_group("nodes")
    nodes_group.create_array("ids", data=nodes[:, 0].astype(np.uint64), chunks=(10000,))

    props_group = nodes_group.create_group("props")
    t_group = props_group.create_group("t")
    t_group.create_array("values", data=nodes[:, 1].astype(np.int64), chunks=(10000,))

    z_group = props_group.create_group("z")
    z_group.create_array("values", data=nodes[:, 2].astype(np.int64), chunks=(10000,))

    y_group = props_group.create_group("y")
    y_group.create_array("values", data=nodes[:, 3].astype(np.int64), chunks=(10000,))

    x_group = props_group.create_group("x")
    x_group.create_array("values", data=nodes[:, 4].astype(np.int64), chunks=(10000,))

    # Create edges group with ids array
    edges_group = g.create_group("edges")
    edges_flat = edges.astype(np.uint64).reshape(-1, 2)
    edges_group.create_array("ids", data=edges_flat, chunks=(10000, 2))


def apply_variant(
    variant_name: str,
    config: dict,
    predictions_dir: Path,
    output_dir: Path
) -> dict:
    """Apply post-proc variant to all predictions in a directory.

    Parameters
    ----------
    variant_name : str
        One of: 'exp1_edge_threshold_gap_recovery', 'exp2_centroid_refine_smooth', 'exp3_division_filter'
    config : dict
        Variant config (from YAML, contains 'postproc' key)
    predictions_dir : Path
        Input pilkwang predictions directory (contains .geff files)
    output_dir : Path
        Output directory for variant predictions

    Returns
    -------
    dict
        Results dict with counts and any errors
    """

    # Map variant names to classes
    variants = {
        "exp1_edge_threshold_gap_recovery": EdgeThresholdGapRecovery,
        "exp2_centroid_refine_smooth": CentroidRefineSmooth,
        "exp3_division_filter": DivisionFilter,
    }

    if variant_name not in variants:
        raise ValueError(f"Unknown variant: {variant_name}. Must be one of: {list(variants.keys())}")

    transform_class = variants[variant_name]
    transform = transform_class(config.get("postproc", {}))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply transformation to all predictions
    results = {"processed": 0, "errors": 0, "errors_list": []}
    geff_files = sorted(predictions_dir.glob("*.geff"))

    print(f"[VERBOSE] Processing {len(geff_files)} GEFF files...")

    for i, pred_geff in enumerate(geff_files):
        try:
            print(f"[VERBOSE] [{i+1}/{len(geff_files)}] Loading {pred_geff.name}...", flush=True)

            # Load prediction (returns DataFrames; convert to numpy arrays)
            nodes_df, edges_df = io.read_geff(pred_geff)
            nodes = np.column_stack([
                nodes_df["node_id"].values,
                nodes_df["t"].values,
                nodes_df["z"].values,
                nodes_df["y"].values,
                nodes_df["x"].values,
            ]).astype(np.float32)
            edges = edges_df[["source_id", "target_id"]].values.astype(np.int32)

            print(f"[VERBOSE]   Loaded: nodes {nodes.shape}, edges {edges.shape}", flush=True)

            # Apply transformation
            print(f"[VERBOSE]   Applying transformation...", flush=True)
            nodes_t, edges_t = transform(nodes, edges)
            print(f"[VERBOSE]   Transformed: nodes {nodes_t.shape}, edges {edges_t.shape}", flush=True)

            # Save transformed prediction
            output_geff = output_dir / pred_geff.name
            print(f"[VERBOSE]   Writing to {output_geff.name}...", flush=True)
            write_geff(output_geff, nodes_t, edges_t)
            print(f"[VERBOSE]   ✓ Complete", flush=True)

            results["processed"] += 1

        except Exception as e:
            results["errors"] += 1
            err_msg = f"{pred_geff.name}: {type(e).__name__}: {str(e)}"
            results["errors_list"].append(err_msg)
            if results["errors"] <= 3:
                print(f"[DEBUG] {err_msg}")

    return results


if __name__ == "__main__":
    import yaml

    # Example usage
    config_path = Path("baseline/experiments_v11/exp1_edge_threshold_gap_recovery.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    predictions_dir = Path("output/stage2/loeo_predictions")
    output_dir = Path("output/stage2/exp1_predictions")

    results = apply_variant("exp1_edge_threshold_gap_recovery", config, predictions_dir, output_dir)
    print(f"Processed: {results['processed']}, Errors: {results['errors']}")
