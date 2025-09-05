# /core/worldgen/world_components/world_graph_navigator.py

from core.common import file_io
from core.common.config_loader import config
from ..procgen_utils import get_relationships_data


class WorldGraphNavigator:
    """
    A read-only service for traversing and querying the in-memory world graph.
    It understands the nested dictionary structure and relationship rules.
    """

    def __init__(self):
        self.RELATIONSHIPS_DATA = get_relationships_data()
        self.HIERARCHICAL_RELATIONSHIPS = self.RELATIONSHIPS_DATA.get("hierarchical", [])
        self.LATTICE_RELATIONSHIPS = self.RELATIONSHIPS_DATA.get("lattice", {})
        self.VALID_RELATIONSHIPS = self.HIERARCHICAL_RELATIONSHIPS + list(self.LATTICE_RELATIONSHIPS.keys())

    def get_available_relationships(self, breadcrumb: list, current_node: dict) -> list[str]:
        """Returns a list of logically possible relationship types from the current location."""
        # Root nodes cannot have lattice relationships.
        if len(breadcrumb) <= 1:
            return self.HIERARCHICAL_RELATIONSHIPS

        occupied_lattice_rels = set()
        # Check explicit relationships stored on the node
        for rel in current_node.get("relationships", {}):
            if rel in self.LATTICE_RELATIONSHIPS:
                occupied_lattice_rels.add(rel)

        available_lattice = [rel for rel in self.LATTICE_RELATIONSHIPS if rel not in occupied_lattice_rels]

        return self.HIERARCHICAL_RELATIONSHIPS + available_lattice

    def find_breadcrumb_for_node(self, target_name: str, current_data=None, current_breadcrumb=None,
                                 visited=None) -> list | None:
        """Recursively searches the world data for a node by name and returns its breadcrumb, now with cycle detection."""
        if current_data is None: current_data = config.world
        if current_breadcrumb is None: current_breadcrumb = []
        if visited is None: visited = set()

        for key, value in current_data.items():
            if key in visited:
                continue

            if isinstance(value, dict):
                visited.add(key)
                new_breadcrumb = current_breadcrumb + [key]
                if value.get("Name") == target_name:
                    return new_breadcrumb

                found = self.find_breadcrumb_for_node(target_name, value.get("Children"), new_breadcrumb, visited)
                if found:
                    return found
        return None

    def build_descriptive_breadcrumb_trail(self, breadcrumb_list: list) -> str:
        """Traverses the world data to build a descriptive path string using Type."""
        if not breadcrumb_list:
            return "Top Level"
        parts = []
        for i in range(len(breadcrumb_list)):
            node = file_io._find_node_by_breadcrumb(config.world, breadcrumb_list[:i + 1])
            if not node:
                parts.append(f"{breadcrumb_list[i]} (Unknown Location)")
                break
            type_str = node.get("Type", "Location")
            parts.append(f"{breadcrumb_list[i]} ({type_str})")
        return " -> ".join(parts)

    def calculate_new_breadcrumb(self, old_breadcrumb: list, relationship: str, target_name: str) -> list | None:
        if relationship in self.HIERARCHICAL_RELATIONSHIPS:
            return old_breadcrumb + [target_name]
        if relationship in self.LATTICE_RELATIONSHIPS:
            return old_breadcrumb[:-1] + [target_name] if old_breadcrumb else [target_name]
        if relationship == "OUTSIDE":
            return old_breadcrumb[:-1] if len(old_breadcrumb) > 1 else old_breadcrumb
        return None