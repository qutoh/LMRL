# /core/ui/views/world_graph_view.py

import random
import time

import tcod
from typing import Dict, Any

from ..ui_framework import View, DynamicTextBox
from ...worldgen.world_components.world_graph_navigator import WorldGraphNavigator
from ...common.config_loader import config


class WorldGraphView(View):
    """A view that visualizes the world data as a node graph."""

    def __init__(self, world_data: Dict[str, Any], console_width: int, console_height: int,
                 tileset: tcod.tileset.Tileset):
        super().__init__()
        self.world_data = world_data
        self.console_width = console_width
        self.console_height = console_height
        self.tileset = tileset
        self.nodes = {}
        self.edges = []
        self.navigator = WorldGraphNavigator()
        self.start_time = time.time()
        self.sdl_primitives = []

        # The navigator relies on the global config.world, so we set it here
        # to ensure it can find nodes for lattice relationship lookups.
        config.world = self.world_data
        self._parse_world_data()
        self._layout_nodes()

    def _add_primitive(self, prim: dict):
        self.sdl_primitives.append(prim)

    def _parse_world_data(self):
        """Recursively traverses the world dictionary to build node and edge lists."""
        self.nodes = {}
        self.edges = []

        def recurse_graph(node_dict, parent_breadcrumb=None):
            for name, data in node_dict.items():
                if not isinstance(data, dict): continue

                breadcrumb = parent_breadcrumb + [name] if parent_breadcrumb else [name]
                breadcrumb_str = "->".join(breadcrumb)

                if breadcrumb_str not in self.nodes:
                    self.nodes[breadcrumb_str] = {
                        'name': name, 'type': data.get('Type', 'N/A'),
                        'description': data.get('Description', 'N/A'),
                        'breadcrumb': breadcrumb, 'x': 0.0, 'y': 0.0
                    }

                if parent_breadcrumb:
                    parent_breadcrumb_str = "->".join(parent_breadcrumb)
                    self.edges.append({
                        'source': parent_breadcrumb_str, 'target': breadcrumb_str,
                        'type': data.get('relationships', {}).get('PARENT_RELATION', 'INSIDE')
                    })

                if 'relationships' in data:
                    for rel, target_name in data['relationships'].items():
                        if rel not in ['PARENT', 'PARENT_RELATION']:
                            target_bc_list = self.navigator.find_breadcrumb_for_node(target_name)
                            if target_bc_list:
                                target_bc_str = "->".join(target_bc_list)
                                self.edges.append({'source': breadcrumb_str, 'target': target_bc_str, 'type': rel})

                if "Children" in data and data["Children"]:
                    recurse_graph(data["Children"], breadcrumb)

        recurse_graph(self.world_data)

    def _layout_nodes(self):
        """Positions nodes using a more stable, physics-based force-directed algorithm."""
        if not self.nodes: return

        node_list = list(self.nodes.values())

        for node in node_list:
            node['x'] = random.uniform(10, self.console_width - 10)
            node['y'] = random.uniform(5, self.console_height - 5)
            node['vx'] = 0.0
            node['vy'] = 0.0

        iterations = 100
        k_repel = 200.0
        k_attract = 0.02
        k_gravity = 0.01
        damping = 0.80

        node_map = {"->".join(n['breadcrumb']): n for n in node_list}
        center_x, center_y = self.console_width / 2.0, self.console_height / 2.0

        for i in range(iterations):
            for n1 in node_list:
                n1['vx'] += (center_x - n1['x']) * k_gravity
                n1['vy'] += (center_y - n1['y']) * k_gravity

                for n2 in node_list:
                    if n1 is n2: continue
                    dx, dy = n1['x'] - n2['x'], n1['y'] - n2['y']
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < 1.0: dist_sq = 1.0

                    dist = dist_sq ** 0.5
                    force = k_repel / dist_sq

                    if dist > 0:
                        n1['vx'] += force * (dx / dist)
                        n1['vy'] += force * (dy / dist)

            for edge in self.edges:
                source = node_map.get(edge['source'])
                target = node_map.get(edge['target'])
                if source and target:
                    dx, dy = target['x'] - source['x'], target['y'] - source['y']
                    source['vx'] += dx * k_attract
                    source['vy'] += dy * k_attract
                    target['vx'] -= dx * k_attract
                    target['vy'] -= dy * k_attract

            for node in node_list:
                node['vx'] *= damping
                node['vy'] *= damping
                node['x'] += node['vx']
                node['y'] += node['vy']
                node['x'] = max(10, min(self.console_width - 10, node['x']))
                node['y'] = max(5, min(self.console_height - 5, node['y']))

    def render(self, console: tcod.console.Console):
        # This method now prepares primitives for the UIManager, it does not render directly.
        self.sdl_primitives.clear()

        # --- PHASE 1: Render status text to the main console ---
        elapsed = time.time() - self.start_time
        status_text = "Generating World" + "." * (int(elapsed * 2) % 4)
        console.print(self.console_width // 2, 1, status_text, fg=(255, 255, 255), alignment=tcod.CENTER)
        console.print(self.console_width // 2, 2, "Press ESC to skip exploration phase", fg=(200, 200, 100),
                      alignment=tcod.CENTER)

        # --- PHASE 2: Prepare line primitives ---
        for edge in self.edges:
            source_node = self.nodes.get(edge['source'])
            target_node = self.nodes.get(edge['target'])
            if not source_node or not target_node: continue

            x1, y1 = int(source_node['x']), int(source_node['y'])
            x2, y2 = int(target_node['x']), int(target_node['y'])

            px1 = x1 * self.tileset.tile_width + self.tileset.tile_width // 2
            py1 = y1 * self.tileset.tile_height + self.tileset.tile_height // 2
            px2 = x2 * self.tileset.tile_width + self.tileset.tile_width // 2
            py2 = y2 * self.tileset.tile_height + self.tileset.tile_height // 2

            is_lattice = edge['type'] in self.navigator.LATTICE_RELATIONSHIPS or edge['type'] == "NEARBY"
            color = (200, 200, 200) if is_lattice else (100, 100, 100)
            self._add_primitive({'type': 'line', 'start': (px1, py1), 'end': (px2, py2), 'color': color})

        # --- PHASE 3: Prepare console primitives for nodes ---
        for breadcrumb, node in self.nodes.items():
            x, y = int(node['x']), int(node['y'])

            # Create a temporary textbox to calculate size and content
            node_text = f"({node['type']})\n{node.get('description', 'No description.')}"
            node_box = DynamicTextBox(
                title=node['name'], text=node_text,
                max_width=35, max_height=8
            )

            if node_box.width <= 0 or node_box.height <= 0: continue

            # Create a small, off-screen console for this node
            temp_console = tcod.console.Console(node_box.width, node_box.height, order="F")

            # Render the textbox onto our temporary console
            node_box.render(temp_console)

            # Calculate pixel position for the top-left corner of the texture
            pixel_x = (x - node_box.width // 2) * self.tileset.tile_width
            pixel_y = (y - node_box.height // 2) * self.tileset.tile_height

            # Add the prepared console and its position as a primitive
            self._add_primitive({'type': 'console', 'console': temp_console, 'x': pixel_x, 'y': pixel_y})