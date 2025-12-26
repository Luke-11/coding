from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set

try:
    import pandas as pd
    HAS_PANDAS = True
except (ImportError, ValueError, Exception):
    HAS_PANDAS = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except (ImportError, ValueError, Exception):
    HAS_NETWORKX = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except (ImportError, ValueError, Exception):
    HAS_PLOTLY = False

@dataclass
class DependencyEntry:
    """
    Represents a single file dependency in a LaTeX project.

    This dataclass stores information about files included or referenced
    in a LaTeX document, including the type of inclusion and location.

    Attributes:
        file_path: The path to the dependent file (relative to project root).
        include_type: The type of inclusion (e.g., "input", "include", "usepackage").
        included_from: The file that includes this dependency.
        line_number: The line number in the including file where the inclusion occurs.
        exists: Whether the file actually exists on disk.
        is_tex_file: Whether this is a .tex file (vs .sty, .cls, etc.).

    Examples:
        >>> entry = DependencyEntry(
        ...     file_path="chapters/ch01.tex",
        ...     include_type="include",
        ...     included_from="main.tex",
        ...     line_number=10,
        ...     exists=True,
        ...     is_tex_file=True
        ... )
    """
    file_path: str
    include_type: str
    included_from: str
    line_number: int
    exists: bool
    is_tex_file: bool

class LatexDependencyCollector:
    """
    Collect information about all files included in a LaTeX project build dependency tree.

    This class parses LaTeX source files to identify all dependencies including:
    - \\input commands
    - \\include commands
    - \\usepackage commands
    - \\documentclass commands
    - Other file references

    The collector builds a complete dependency tree starting from a root file
    and recursively following all inclusions.

    Attributes:
        base_path (Path): Base directory path where the LaTeX project is located.
        dependencies (List[DependencyEntry]): List of all dependency entries found.
        processed_files (Set[str]): Set of files that have already been processed
                                   to avoid infinite recursion.

    Regex patterns:
        INPUT_RE: Matches \\input{filename} and \\input filename
        INCLUDE_RE: Matches \\include{filename}
        USEPACKAGE_RE: Matches \\usepackage[options]{package}
        DOCUMENTCLASS_RE: Matches \\documentclass[options]{class}
        BIB_RE: Matches \\bibliography{file} and \\addbibresource{file}
    """

    # Regex patterns for different types of file inclusions
    INPUT_RE = re.compile(r"""\\input\s*(?:\{([^}]+)\}|(\S+))""")
    INCLUDE_RE = re.compile(r"""\\include\s*\{([^}]+)\}""")
    USEPACKAGE_RE = re.compile(r"""\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}""")
    DOCUMENTCLASS_RE = re.compile(r"""\\documentclass\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}""")
    BIB_RE = re.compile(r"""\\(?:bibliography|addbibresource)\s*\{([^}]+)\}""")

    def __init__(self, base_path: Path | str):
        """
        Initialize the dependency collector with a base path.

        Args:
            base_path: Base directory path where the LaTeX project files are located.
                      Can be a string or Path object.

        Raises:
            TypeError: If base_path cannot be converted to a Path.
        """
        self.base_path = Path(base_path)
        self.dependencies: List[DependencyEntry] = []
        self.processed_files: Set[str] = set()

    def _normalize_file_path(self, file_path: str, from_file: str) -> str:
        """
        Normalize a file path relative to the project root.

        Handles relative paths, extensions, and resolves paths based on LaTeX conventions.
        In LaTeX, \\input and \\include paths are typically relative to the main document
        directory, not the including file's directory.

        Args:
            file_path: The raw file path from the LaTeX command.
            from_file: The file that contains the inclusion (for context only).

        Returns:
            Normalized path relative to base_path.
        """
        # Handle common LaTeX file extensions
        if not Path(file_path).suffix:
            # Add .tex for input/include if no extension
            file_path += '.tex'

        # Convert to Path object
        path = Path(file_path)

        # If absolute path, try to make relative to base_path
        if path.is_absolute():
            try:
                return str(path.relative_to(self.base_path))
            except ValueError:
                return str(path)

        # For LaTeX, paths in \\input and \\include are typically relative to the main document directory
        # (where the root .tex file is), not the including file's directory
        resolved = self.base_path / path

        # Try to make relative to base_path
        try:
            return str(resolved.relative_to(self.base_path))
        except ValueError:
            return str(resolved)

    def _parse_file_dependencies(self, file_path: str) -> List[Tuple[str, str, int]]:
        """
        Parse a single file for dependencies.

        Reads the file and extracts all file inclusions using regex patterns.

        Args:
            file_path: Path to the file to parse (relative to base_path).

        Returns:
            List of tuples (include_type, included_file, line_number).
        """
        full_path = self.base_path / file_path
        if not full_path.exists():
            return []

        dependencies = []
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    # Check each type of inclusion
                    for pattern_name, pattern in [
                        ('input', self.INPUT_RE),
                        ('include', self.INCLUDE_RE),
                        ('usepackage', self.USEPACKAGE_RE),
                        ('documentclass', self.DOCUMENTCLASS_RE),
                        ('bibliography', self.BIB_RE)
                    ]:
                        matches = pattern.findall(line)
                        for match in matches:
                            # Handle different capture group structures
                            if isinstance(match, tuple):
                                # For patterns with multiple capture groups (like input)
                                dep_file = match[0] or match[1]
                            else:
                                # For patterns with single capture group
                                dep_file = match

                            if dep_file:
                                dependencies.append((pattern_name, dep_file.strip(), line_num))
        except Exception:
            # Skip files that can't be read
            pass

        return dependencies

    def _collect_dependencies_recursive(self, file_path: str, depth: int = 0) -> None:
        """
        Recursively collect dependencies starting from a file.

        Args:
            file_path: Path to the file to process (relative to base_path).
            depth: Current recursion depth (for cycle detection).
        """
        # Avoid infinite recursion
        if depth > 50 or file_path in self.processed_files:
            return

        self.processed_files.add(file_path)

        # Parse dependencies in this file
        deps = self._parse_file_dependencies(file_path)

        for include_type, raw_dep_path, line_num in deps:
            # Normalize the path
            normalized_path = self._normalize_file_path(raw_dep_path, file_path)

            # Check if file exists
            full_path = self.base_path / normalized_path
            exists = full_path.exists()
            is_tex = full_path.suffix.lower() == '.tex'

            # Create dependency entry
            entry = DependencyEntry(
                file_path=normalized_path,
                include_type=include_type,
                included_from=file_path,
                line_number=line_num,
                exists=exists,
                is_tex_file=is_tex
            )

            self.dependencies.append(entry)

            # Recursively process .tex files
            if exists and is_tex and include_type in ('input', 'include'):
                self._collect_dependencies_recursive(normalized_path, depth + 1)

    def collect_dependencies(self, root_file: str = "main.tex") -> List[DependencyEntry]:
        """
        Collect all dependencies starting from a root file.

        This is the main method that builds the complete dependency tree
        by recursively parsing all included files.

        Args:
            root_file: The root LaTeX file to start from (relative to base_path).
                      Defaults to "main.tex".

        Returns:
            List of DependencyEntry objects representing all dependencies found.

        Examples:
            >>> collector = LatexDependencyCollector("../latex_split_test_project")
            >>> deps = collector.collect_dependencies("main.tex")
            >>> len(deps)
            15
            >>> tex_files = [d for d in deps if d.is_tex_file and d.exists]
        """
        # Reset state
        self.dependencies = []
        self.processed_files = set()

        # Start recursive collection
        self._collect_dependencies_recursive(root_file)

        return self.dependencies

    def get_dependency_tree(self, root_file: str = "main.tex") -> Dict[str, List[DependencyEntry]]:
        """
        Get dependencies organized as a tree structure.

        Returns a dictionary where keys are file paths and values are lists
        of files they depend on.

        Args:
            root_file: The root LaTeX file to start from (relative to base_path).

        Returns:
            Dictionary mapping file paths to their direct dependencies.
        """
        deps = self.collect_dependencies(root_file)
        tree = {}

        for dep in deps:
            if dep.included_from not in tree:
                tree[dep.included_from] = []
            tree[dep.included_from].append(dep)

        return tree

    def get_all_files(self, root_file: str = "main.tex") -> Set[str]:
        """
        Get all files involved in the build (including the root file).

        Args:
            root_file: The root LaTeX file to start from (relative to base_path).

        Returns:
            Set of all file paths involved in the build.
        """
        deps = self.collect_dependencies(root_file)
        files = set([root_file])
        files.update(dep.file_path for dep in deps)
        return files

    def to_dataframe(self, root_file: str = "main.tex"):
        """
        Collect dependencies and return as a pandas DataFrame.

        Args:
            root_file: The root LaTeX file to start from (relative to base_path).

        Returns:
            pandas DataFrame with columns: file_path, include_type, included_from,
            line_number, exists, is_tex_file. Returns None if pandas not available.
        """
        if not HAS_PANDAS:
            return None

        deps = self.collect_dependencies(root_file)
        data = [{
            'file_path': d.file_path,
            'include_type': d.include_type,
            'included_from': d.included_from,
            'line_number': d.line_number,
            'exists': d.exists,
            'is_tex_file': d.is_tex_file
        } for d in deps]

        return pd.DataFrame(data)

    def print_table(self, root_file: str = "main.tex") -> None:
        """
        Collect dependencies and print as a formatted table.

        Args:
            root_file: The root LaTeX file to start from (relative to base_path).

        Returns:
            None. Output is printed to stdout.
        """
        deps = self.collect_dependencies(root_file)

        if not deps:
            print("No dependencies found")
            return

        if HAS_PANDAS:
            df = self.to_dataframe(root_file)
            if df is not None:
                print(df.to_string(index=False))
        else:
            # Fallback table printing
            headers = ['file_path', 'include_type', 'included_from', 'line_number', 'exists', 'is_tex_file']
            col_widths = {h: max(len(str(getattr(d, h))) for d in deps +
                                 [type('Mock', (), {h: h for h in headers})()]) for h in headers}

            # Print header
            header_row = ' | '.join(f"{h:<{col_widths[h]}}" for h in headers)
            print(header_row)
            print('-' * len(header_row))

            # Print data rows
            for d in deps:
                row = [str(getattr(d, h)) for h in headers]
                print(' | '.join(f"{cell:<{col_widths[h]}}" for cell, h in zip(row, headers)))

    def to_networkx_graph(self, root_file: str = "main.tex") -> Optional["nx.DiGraph"]:
        """
        Create a NetworkX directed graph representation of the dependency tree.

        Args:
            root_file: The root LaTeX file to start from (relative to base_path).

        Returns:
            A NetworkX DiGraph where nodes are files and edges represent dependencies.
            Returns None if networkx is not available.

        Note:
            Node attributes include:
            - 'exists': Whether the file exists
            - 'is_tex_file': Whether it's a .tex file
            - 'node_type': 'root', 'tex_file', or 'other_file'

            Edge attributes include:
            - 'include_type': Type of inclusion ('input', 'include', 'usepackage', etc.)
            - 'line_number': Line number where the inclusion occurs
        """
        if not HAS_NETWORKX:
            return None

        deps = self.collect_dependencies(root_file)
        G = nx.DiGraph()

        # Add root node
        G.add_node(root_file, exists=True, is_tex_file=True, node_type='root')

        # Add all dependency nodes and edges
        for dep in deps:
            # Add the included_from node if not already present
            if dep.included_from not in G:
                # Check if it's a tex file by looking at existing dependencies
                is_tex = any(d.file_path == dep.included_from and d.is_tex_file for d in deps)
                is_tex = is_tex or dep.included_from.endswith('.tex')
                G.add_node(dep.included_from, exists=True, is_tex_file=is_tex, node_type='tex_file')

            # Add the dependency node
            node_type = 'tex_file' if dep.is_tex_file else 'other_file'
            G.add_node(dep.file_path, exists=dep.exists, is_tex_file=dep.is_tex_file, node_type=node_type)

            # Add the edge
            G.add_edge(dep.included_from, dep.file_path,
                      include_type=dep.include_type,
                      line_number=dep.line_number)

        return G

    def visualize_dependency_graph(self, root_file: str = "main.tex", layout: str = "spring") -> Optional["go.Figure"]:
        """
        Create an interactive plotly visualization of the dependency graph.

        Args:
            root_file: The root LaTeX file to start from (relative to base_path).
            layout: Layout algorithm to use ('spring', 'circular', 'random', 'shell', 'kamada_kawai').

        Returns:
            A plotly Figure object that can be displayed or saved.
            Returns None if networkx or plotly are not available.

        Examples:
            >>> collector = LatexDependencyCollector("../latex_split_test_project")
            >>> fig = collector.visualize_dependency_graph("main.tex")
            >>> fig.show()  # Display in browser
            >>> fig.write_html("dependency_graph.html")  # Save as HTML
        """
        if not HAS_NETWORKX or not HAS_PLOTLY:
            print("NetworkX and Plotly are required for graph visualization")
            return None

        G = self.to_networkx_graph(root_file)
        if G is None or len(G.nodes) == 0:
            print("No graph data available")
            return None

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "random":
            pos = nx.random_layout(G, seed=42)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_text = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Create hover text for edge
            edge_data = edge[2]
            hover_text = f"From: {edge[0]}<br>To: {edge[1]}<br>Type: {edge_data.get('include_type', 'unknown')}<br>Line: {edge_data.get('line_number', 'unknown')}"
            edge_text.append(hover_text)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines',
            name='dependencies'
        )

        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []

        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)

            # Node information
            node_data = node[1]
            node_type = node_data.get('node_type', 'other_file')
            exists = node_data.get('exists', False)
            is_tex = node_data.get('is_tex_file', False)

            # Color coding
            if node_type == 'root':
                color = '#FF6B6B'  # Red for root
                size = 30
            elif node_type == 'tex_file':
                color = '#4ECDC4' if exists else '#95A5A6'  # Teal for tex files, gray for missing
                size = 25
            else:
                color = '#45B7D1' if exists else '#95A5A6'  # Blue for other files, gray for missing
                size = 20

            node_colors.append(color)
            node_sizes.append(size)

            # Hover text
            hover_text = f"File: {node[0]}<br>Type: {node_type}<br>Exists: {exists}<br>Is TeX: {is_tex}"
            node_text.append(hover_text)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node[0].split('/')[-1] for node in G.nodes()],  # Show just filename
            textposition="bottom center",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line_width=2,
                line_color='white'
            ),
            name='files'
        )

        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f"LaTeX Dependency Graph - {root_file}",
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))

        return fig

    def print_graph_info(self, root_file: str = "main.tex") -> None:
        """
        Print basic information about the dependency graph.

        Args:
            root_file: The root LaTeX file to analyze.

        Returns:
            None. Information is printed to stdout.
        """
        if not HAS_NETWORKX:
            print("NetworkX is required for graph analysis")
            return

        G = self.to_networkx_graph(root_file)
        if G is None:
            print("No graph data available")
            return

        print("LaTeX Dependency Graph Information")
        print("=" * 40)
        print(f"Root file: {root_file}")
        print(f"Total nodes (files): {len(G.nodes)}")
        print(f"Total edges (dependencies): {len(G.edges)}")

        # Count node types
        node_types = {}
        for node, data in G.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print("\nNode types:")
        for node_type, count in node_types.items():
            print(f"  {node_type}: {count}")

        # Count edge types
        edge_types = {}
        for _, _, data in G.edges(data=True):
            edge_type = data.get('include_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        print("\nDependency types:")
        for edge_type, count in sorted(edge_types.items()):
            print(f"  {edge_type}: {count}")

        # Graph properties
        if nx.is_directed(G):
            print(f"\nGraph is directed: Yes")
            print(f"Is DAG (no cycles): {nx.is_directed_acyclic_graph(G)}")

            # Find root and leaves
            roots = [node for node in G.nodes() if G.in_degree(node) == 0]
            leaves = [node for node in G.nodes() if G.out_degree(node) == 0]

            print(f"Root nodes: {len(roots)} - {roots[:3]}{'...' if len(roots) > 3 else ''}")
            print(f"Leaf nodes: {len(leaves)} - {leaves[:3]}{'...' if len(leaves) > 3 else ''}")

        print(f"\nGraph density: {nx.density(G):.4f}")

    def save_dependency_graph_clean(self, root_file: str = "main.tex", output_file: str = "dependency_graph.html") -> bool:
        """
        Save the dependency graph as a clean, readable HTML file with separate JS data.

        Args:
            root_file: The root LaTeX file to start from (relative to base_path).
            output_file: The output HTML file path.

        Returns:
            True if successful, False otherwise.
        """
        if not HAS_NETWORKX or not HAS_PLOTLY:
            print("NetworkX and Plotly are required for graph visualization")
            return False

        fig = self.visualize_dependency_graph(root_file)
        if fig is None:
            return False

        # Convert figure to JSON
        fig_json = fig.to_json()

        # Create clean HTML template
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LaTeX Dependency Graph - {root_file}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        #graph {{
            width: 100%;
            height: 800px;
        }}
        .info {{
            background: #f8f9fa;
            border-left: 4px solid #007acc;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LaTeX Dependency Graph</h1>
        <p class="subtitle">Interactive visualization of file dependencies for {root_file}</p>

        <div class="info">
            <strong>Graph Statistics:</strong><br>
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            Root file: {root_file}
        </div>

        <div class="legend">
            <div class="legend-item">
                <span class="legend-color" style="background-color: #FF6B6B;"></span>
                <span>Root File</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #4ECDC4;"></span>
                <span>TeX Files (Existing)</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #95A5A6;"></span>
                <span>Missing Files</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #45B7D1;"></span>
                <span>Other Files</span>
            </div>
        </div>

        <div id="graph"></div>
    </div>

    <script>
        // Figure data
        const figureData = {fig_json};

        // Render the plot
        Plotly.newPlot('graph', figureData.data, figureData.layout, {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
            displaylogo: false
        }});
    </script>
</body>
</html>"""

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_template)
            print(f"Clean HTML graph saved as '{output_file}'")
            return True
        except Exception as e:
            print(f"Error saving HTML file: {e}")
            return False

    def save_dependency_graph_separate(self, root_file: str = "main.tex",
                                     html_file: str = "dependency_graph.html",
                                     data_file: str = "dependency_graph.json") -> bool:
        """
        Save the dependency graph as separate HTML and JSON files for better maintainability.

        This creates:
        1. A clean HTML template that loads the visualization
        2. A separate JSON file containing the graph data

        Args:
            root_file: The root LaTeX file to start from (relative to base_path).
            html_file: The output HTML file path.
            data_file: The output JSON data file path.

        Returns:
            True if successful, False otherwise.

        Benefits:
        - HTML template is reusable for different graphs
        - Data is separate and can be version controlled independently
        - Smaller, more maintainable files
        - Easy to update visualization without touching data
        """
        if not HAS_NETWORKX or not HAS_PLOTLY:
            print("NetworkX and Plotly are required for graph visualization")
            return False

        fig = self.visualize_dependency_graph(root_file)
        if fig is None:
            return False

        # Convert figure to JSON
        fig_json = fig.to_json()

        # Save the data file
        try:
            with open(data_file, 'w', encoding='utf-8') as f:
                f.write(fig_json)
            print(f"Graph data saved as '{data_file}'")
        except Exception as e:
            print(f"Error saving data file: {e}")
            return False

        # Create clean HTML template that embeds the data
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LaTeX Dependency Graph - {root_file}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        #graph {{
            width: 100%;
            height: 800px;
        }}
        .info {{
            background: #f8f9fa;
            border-left: 4px solid #007acc;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LaTeX Dependency Graph</h1>
        <p class="subtitle">Interactive visualization of file dependencies for {root_file}</p>

        <div class="info">
            <strong>Graph Statistics:</strong><br>
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            Root file: {root_file}<br>
            Data source: {data_file}
        </div>

        <div class="legend">
            <div class="legend-item">
                <span class="legend-color" style="background-color: #FF6B6B;"></span>
                <span>Root File</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #4ECDC4;"></span>
                <span>TeX Files (Existing)</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #95A5A6;"></span>
                <span>Missing Files</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #45B7D1;"></span>
                <span>Other Files</span>
            </div>
        </div>

        <div id="graph"></div>
    </div>

    <script type="application/json" id="graph-data">
{fig_json}
    </script>

    <script>
        // Load graph data from embedded JSON
        const figureData = JSON.parse(document.getElementById('graph-data').textContent);

        // Render the plot
        Plotly.newPlot('graph', figureData.data, figureData.layout, {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
            displaylogo: false
        }});
    </script>
</body>
</html>"""

        # Save the HTML file
        try:
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_template)
            print(f"HTML template saved as '{html_file}'")
            return True
        except Exception as e:
            print(f"Error saving HTML file: {e}")
            return False
