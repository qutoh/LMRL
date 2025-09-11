# /core/worldgen/v3_components/converter.py

from typing import List, Dict

from core.common import file_io
from core.common.game_state import LayoutGraph, GenerationState
from .feature_node import FeatureNode


class Converter:
    """
    Handles final data conversion from the internal FeatureNode representation
    into formats required by the rest of the application (GenerationState, LayoutGraph).
    """

    @staticmethod
    def populate_generation_state(gen_state: GenerationState, all_branches: List[FeatureNode]):
        gen_state.placed_features.clear()
        all_nodes = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]

        for i, node in enumerate(all_nodes):
            tag = file_io.sanitize_filename(node.name)
            if tag in gen_state.placed_features:
                tag = f"{tag}_{i}"

            feature_dict = {
                "name": node.name,
                "type": node.feature_type,
                "bounding_box": node.get_rect(),
                "footprint": [list(p) for p in node.get_absolute_footprint()],
                "interior_footprint": [list(p) for p in node.get_absolute_interior_footprint()],
                "narrative_tag": node.name
            }
            if node.path_coords:
                feature_dict["path_tiles"] = node.path_coords
            if node.tile_overrides:
                feature_dict["tile_overrides"] = {f"{x},{y}": v for (x, y), v in node.tile_overrides.items()}

            gen_state.placed_features[tag] = feature_dict
    # ... (rest of converter.py is unchanged)
    @staticmethod
    def serialize_feature_tree_to_graph(all_branches: List[FeatureNode]) -> LayoutGraph:
        """Converts the internal FeatureNode tree into a serializable LayoutGraph."""
        graph = LayoutGraph()
        all_nodes = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]
        node_to_tag_map = {}

        for i, node in enumerate(all_nodes):
            tag = file_io.sanitize_filename(node.name)
            if tag in graph.nodes:
                tag = f"{tag}_{i}"
            node_to_tag_map[node] = tag

            node_data = {"name": node.name, "type": node.feature_type, "bounding_box": node.get_rect()}
            if node.path_coords:
                node_data["path_tiles"] = node.path_coords
            graph.nodes[tag] = node_data

        for node in all_nodes:
            for i, feature_data in enumerate(node.placed_interior_features):
                tag = file_io.sanitize_filename(feature_data['name'])
                if tag in graph.nodes:
                    tag = f"{tag}_interior_{i}"
                processed_data = feature_data.copy()
                if 'bounding_box' in processed_data:
                    bb = processed_data['bounding_box']
                    if len(bb) == 4:
                        x1, y1, x2, y2 = bb
                        processed_data['bounding_box'] = (x1, y1, x2 - x1, y2 - y1)
                graph.nodes[tag] = processed_data

        for node, tag in node_to_tag_map.items():
            if node.parent:
                parent_tag = node_to_tag_map.get(node.parent)
                if parent_tag:
                    graph.edges.append((parent_tag, tag, 'any'))
        return graph

    @staticmethod
    def convert_to_vertex_representation(layout_graph: LayoutGraph, connections: List[Dict]) -> Dict:
        bodies = []
        for tag, feature in layout_graph.nodes.items():
            x, y, w, h = feature.get('bounding_box', (0, 0, 0, 0))
            body = {
                "id": tag,
                "name": feature.get('name'),
                "type": feature.get('type'),
                "shape_vertices": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                "metadata": {"narrative_tag": feature.get('narrative_tag')}
            }
            bodies.append(body)

        joints = []
        name_to_tag_map = {f.get('name'): t for t, f in layout_graph.nodes.items()}

        for i, conn in enumerate(connections):
            tag_a = name_to_tag_map.get(conn['node_a'].name)
            tag_b = name_to_tag_map.get(conn['node_b'].name)

            if tag_a and tag_b:
                joints.append({
                    "id": f"joint_{i}", "body_a": tag_a, "body_b": tag_b,
                    "position": conn['position'], "type": "portal"
                })
        return {"bodies": bodies, "joints": joints}