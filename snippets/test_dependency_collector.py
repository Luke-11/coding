import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List

from dependency_collector import LatexDependencyCollector, DependencyEntry


class TestLatexDependencyCollector:
    """Test suite for LatexDependencyCollector class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_latex_files(self, temp_dir):
        """Create sample LaTeX files with various dependencies."""
        # Create main.tex
        main_tex = temp_dir / "main.tex"
        main_tex.write_text(r"""\documentclass{book}
\usepackage{amsmath}
\usepackage{hyperref}
\input{preamble}
\include{chapters/ch01}
\include{chapters/ch02}
\bibliography{references}
""")

        # Create preamble.tex
        preamble_tex = temp_dir / "preamble.tex"
        preamble_tex.write_text(r"""\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{../snippets/secid}
""")

        # Create chapters directory and files
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir()

        ch01_tex = chapters_dir / "ch01.tex"
        ch01_tex.write_text(r"""\chapter{Introduction}
\input{sections/sec01}
\input{sections/sec02}
""")

        ch02_tex = chapters_dir / "ch02.tex"
        ch02_tex.write_text(r"""\chapter{Methods}
Some content here.
""")

        # Create sections directory
        sections_dir = chapters_dir / "sections"
        sections_dir.mkdir()

        sec01_tex = sections_dir / "sec01.tex"
        sec01_tex.write_text(r"""\section{Motivation}
\input{sub01}
\input{sub02}
""")

        sec02_tex = sections_dir / "sec02.tex"
        sec02_tex.write_text(r"""\section{Background}
Content here.
""")

        # Create subsections
        sub01_tex = sections_dir / "sub01.tex"
        sub01_tex.write_text(r"""\subsection{Problem}
Some text.
""")

        sub02_tex = sections_dir / "sub02.tex"
        sub02_tex.write_text(r"""\subsection{Solution}
More text.
""")

        # Create a missing file reference
        missing_tex = temp_dir / "missing.tex"
        # Don't create the file - it should be marked as not existing

        # Create a file with missing dependency
        with_missing_tex = temp_dir / "with_missing.tex"
        with_missing_tex.write_text(r"""\input{missing}
\input{preamble}
""")

        return temp_dir

    @pytest.fixture
    def collector(self, sample_latex_files):
        """Create a LatexDependencyCollector instance."""
        return LatexDependencyCollector(sample_latex_files)

    def test_init(self, temp_dir):
        """Test initialization of LatexDependencyCollector."""
        collector = LatexDependencyCollector(temp_dir)
        assert collector.base_path == Path(temp_dir)
        assert collector.dependencies == []
        assert collector.processed_files == set()

    def test_normalize_file_path_with_extension(self, collector):
        """Test path normalization with existing extension."""
        result = collector._normalize_file_path("chapters/ch01.tex", "main.tex")
        assert result == "chapters/ch01.tex"

    def test_normalize_file_path_without_extension(self, collector):
        """Test path normalization without extension (should add .tex)."""
        result = collector._normalize_file_path("chapters/ch01", "main.tex")
        assert result == "chapters/ch01.tex"

    def test_normalize_file_path_relative(self, collector):
        """Test path normalization with relative paths."""
        result = collector._normalize_file_path("sections/sec01.tex", "chapters/ch01.tex")
        assert result == "sections/sec01.tex"

    def test_parse_file_dependencies_main(self, collector, sample_latex_files):
        """Test parsing dependencies from main.tex."""
        deps = collector._parse_file_dependencies("main.tex")

        # Should find documentclass, usepackage, input, include, bibliography
        expected_deps = [
            ("documentclass", "book", 1),
            ("usepackage", "amsmath", 2),
            ("usepackage", "hyperref", 3),
            ("input", "preamble", 4),
            ("include", "chapters/ch01", 5),
            ("include", "chapters/ch02", 6),
            ("bibliography", "references", 7),
        ]

        assert len(deps) == len(expected_deps)
        for expected in expected_deps:
            assert expected in deps

    def test_parse_file_dependencies_preamble(self, collector, sample_latex_files):
        """Test parsing dependencies from preamble.tex."""
        deps = collector._parse_file_dependencies("preamble.tex")

        expected_deps = [
            ("usepackage", "graphicx", 1),
            ("usepackage", "enumitem", 2),
            ("usepackage", "../snippets/secid", 3),
        ]

        assert len(deps) == len(expected_deps)
        for expected in expected_deps:
            assert expected in deps

    def test_collect_dependencies_main(self, collector):
        """Test collecting all dependencies from main.tex."""
        deps = collector.collect_dependencies("main.tex")

        assert len(deps) > 0

        # Check that we have the expected dependency types
        dep_types = {dep.include_type for dep in deps}
        assert "documentclass" in dep_types
        assert "usepackage" in dep_types
        assert "input" in dep_types
        assert "include" in dep_types
        assert "bibliography" in dep_types

        # Check that main.tex is processed
        assert "main.tex" in collector.processed_files

    def test_collect_dependencies_with_missing_file(self, collector, sample_latex_files):
        """Test collecting dependencies when a file references missing files."""
        deps = collector.collect_dependencies("with_missing.tex")

        # Should have dependencies for missing.tex and preamble.tex
        missing_deps = [d for d in deps if d.file_path == "missing.tex"]
        assert len(missing_deps) == 1
        assert not missing_deps[0].exists
        assert missing_deps[0].is_tex_file  # .tex extension means it's considered a tex file

        preamble_deps = [d for d in deps if d.file_path == "preamble.tex"]
        assert len(preamble_deps) == 1
        assert preamble_deps[0].exists
        assert preamble_deps[0].is_tex_file

    def test_get_dependency_tree(self, collector):
        """Test getting dependencies as a tree structure."""
        tree = collector.get_dependency_tree("main.tex")

        assert isinstance(tree, dict)
        assert "main.tex" in tree

        # main.tex should have multiple dependencies
        main_deps = tree["main.tex"]
        assert len(main_deps) > 0

        # Check that dependencies have correct structure
        for dep in main_deps:
            assert isinstance(dep, DependencyEntry)
            assert dep.included_from == "main.tex"

    def test_get_all_files(self, collector):
        """Test getting all files involved in the build."""
        files = collector.get_all_files("main.tex")

        assert isinstance(files, set)
        assert "main.tex" in files
        assert "preamble.tex" in files
        assert "chapters/ch01.tex" in files

        # Should include files that exist
        existing_files = [f for f in files if (collector.base_path / f).exists]
        assert len(existing_files) > 0

    @pytest.mark.skipif(not hasattr(LatexDependencyCollector, 'HAS_PANDAS') or not LatexDependencyCollector.HAS_PANDAS,
                       reason="pandas not available")
    def test_to_dataframe(self, collector):
        """Test converting dependencies to pandas DataFrame."""
        df = collector.to_dataframe("main.tex")

        assert df is not None
        expected_columns = ['file_path', 'include_type', 'included_from', 'line_number', 'exists', 'is_tex_file']
        assert list(df.columns) == expected_columns
        assert len(df) > 0

    def test_print_table(self, collector, capsys):
        """Test printing dependencies as a table."""
        collector.print_table("main.tex")

        captured = capsys.readouterr()
        assert "file_path" in captured.out
        assert "include_type" in captured.out
        assert len(captured.out.strip()) > 0

    @pytest.mark.skipif(not hasattr(LatexDependencyCollector, 'HAS_NETWORKX') or not LatexDependencyCollector.HAS_NETWORKX,
                       reason="networkx not available")
    def test_to_networkx_graph(self, collector):
        """Test creating NetworkX graph representation."""
        G = collector.to_networkx_graph("main.tex")

        assert G is not None
        assert len(G.nodes) > 0
        assert len(G.edges) > 0

        # Check that root node has correct attributes
        root_attrs = G.nodes["main.tex"]
        assert root_attrs["node_type"] == "root"
        assert root_attrs["exists"] is True
        assert root_attrs["is_tex_file"] is True

    @pytest.mark.skipif(not hasattr(LatexDependencyCollector, 'HAS_NETWORKX') or not LatexDependencyCollector.HAS_NETWORKX,
                       reason="networkx not available")
    @pytest.mark.skipif(not hasattr(LatexDependencyCollector, 'HAS_PLOTLY') or not LatexDependencyCollector.HAS_PLOTLY,
                       reason="plotly not available")
    def test_visualize_dependency_graph(self, collector):
        """Test creating plotly visualization."""
        fig = collector.visualize_dependency_graph("main.tex")

        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        assert len(fig.data) > 0

    @pytest.mark.skipif(not hasattr(LatexDependencyCollector, 'HAS_NETWORKX') or not LatexDependencyCollector.HAS_NETWORKX,
                       reason="networkx not available")
    def test_print_graph_info(self, collector, capsys):
        """Test printing graph information."""
        collector.print_graph_info("main.tex")

        captured = capsys.readouterr()
        assert "LaTeX Dependency Graph Information" in captured.out
        assert "Total nodes" in captured.out
        assert "Total edges" in captured.out

    @pytest.mark.skipif(not hasattr(LatexDependencyCollector, 'HAS_NETWORKX') or not LatexDependencyCollector.HAS_NETWORKX,
                       reason="networkx not available")
    @pytest.mark.skipif(not hasattr(LatexDependencyCollector, 'HAS_PLOTLY') or not LatexDependencyCollector.HAS_PLOTLY,
                       reason="plotly not available")
    def test_save_dependency_graph_clean(self, collector, temp_dir):
        """Test saving clean HTML graph."""
        output_file = temp_dir / "test_graph.html"
        result = collector.save_dependency_graph_clean("main.tex", str(output_file))

        assert result is True
        assert output_file.exists()
        content = output_file.read_text()
        assert "LaTeX Dependency Graph" in content
        assert "Plotly" in content

    @pytest.mark.skipif(not hasattr(LatexDependencyCollector, 'HAS_NETWORKX') or not LatexDependencyCollector.HAS_NETWORKX,
                       reason="networkx not available")
    @pytest.mark.skipif(not hasattr(LatexDependencyCollector, 'HAS_PLOTLY') or not LatexDependencyCollector.HAS_PLOTLY,
                       reason="plotly not available")
    def test_save_dependency_graph_separate(self, collector, temp_dir):
        """Test saving separate HTML and JSON files."""
        html_file = temp_dir / "test_graph.html"
        json_file = temp_dir / "test_graph.json"

        result = collector.save_dependency_graph_separate(
            "main.tex",
            str(html_file),
            str(json_file)
        )

        assert result is True
        assert html_file.exists()
        assert json_file.exists()

        # Check HTML content
        html_content = html_file.read_text()
        assert "LaTeX Dependency Graph" in html_content
        assert "graph-data" in html_content  # Should have embedded JSON

        # Check JSON content
        json_content = json_file.read_text()
        assert '"data"' in json_content
        assert '"layout"' in json_content

    def test_dependency_entry_creation(self, collector):
        """Test DependencyEntry creation and attributes."""
        deps = collector.collect_dependencies("main.tex")

        for dep in deps:
            assert isinstance(dep, DependencyEntry)
            assert isinstance(dep.file_path, str)
            assert isinstance(dep.include_type, str)
            assert isinstance(dep.included_from, str)
            assert isinstance(dep.line_number, int)
            assert isinstance(dep.exists, bool)
            assert isinstance(dep.is_tex_file, bool)

    def test_circular_dependency_prevention(self, collector, sample_latex_files):
        """Test that circular dependencies are prevented."""
        # Create a circular dependency
        ch01_tex = sample_latex_files / "chapters" / "ch01.tex"
        original_content = ch01_tex.read_text()
        ch01_tex.write_text(original_content + "\n\\input{../main}\n")

        # Should not cause infinite recursion
        deps = collector.collect_dependencies("main.tex")

        # Should still complete and have finite dependencies
        assert isinstance(deps, list)
        assert len(deps) < 100  # Reasonable upper bound

    def test_empty_file(self, collector, sample_latex_files):
        """Test handling of empty LaTeX files."""
        empty_tex = sample_latex_files / "empty.tex"
        empty_tex.write_text("")

        deps = collector._parse_file_dependencies("empty.tex")
        assert deps == []

    def test_file_not_found(self, collector):
        """Test handling of non-existent files."""
        deps = collector._parse_file_dependencies("nonexistent.tex")
        assert deps == []

    def test_regex_patterns(self, collector):
        """Test that regex patterns work correctly."""
        # Test input pattern
        test_content = r"\input{file1} \input file2"
        temp_file = collector.base_path / "test_regex.tex"
        temp_file.write_text(test_content)

        deps = collector._parse_file_dependencies("test_regex.tex")
        input_deps = [d for d in deps if d[0] == "input"]
        assert len(input_deps) == 2
        assert ("input", "file1", 1) in deps
        assert ("input", "file2", 1) in deps

    def test_include_pattern(self, collector):
        """Test include pattern matching."""
        test_content = r"\include{chapters/ch01}"
        temp_file = collector.base_path / "test_include.tex"
        temp_file.write_text(test_content)

        deps = collector._parse_file_dependencies("test_include.tex")
        assert len(deps) == 1
        assert deps[0] == ("include", "chapters/ch01", 1)

    def test_usepackage_pattern(self, collector):
        """Test usepackage pattern matching."""
        test_content = r"\usepackage{amsmath} \usepackage[options]{hyperref}"
        temp_file = collector.base_path / "test_pkg.tex"
        temp_file.write_text(test_content)

        deps = collector._parse_file_dependencies("test_pkg.tex")
        usepackage_deps = [d for d in deps if d[0] == "usepackage"]
        assert len(usepackage_deps) == 2
        assert ("usepackage", "amsmath", 1) in deps
        assert ("usepackage", "hyperref", 1) in deps

    def test_documentclass_pattern(self, collector):
        """Test documentclass pattern matching."""
        test_content = r"\documentclass{book} \documentclass[options]{article}"
        temp_file = collector.base_path / "test_doc.tex"
        temp_file.write_text(test_content)

        deps = collector._parse_file_dependencies("test_doc.tex")
        doc_deps = [d for d in deps if d[0] == "documentclass"]
        assert len(doc_deps) == 2
        assert ("documentclass", "book", 1) in deps
        assert ("documentclass", "article", 1) in deps

    def test_bibliography_pattern(self, collector):
        """Test bibliography pattern matching."""
        test_content = r"\bibliography{refs} \addbibresource{main.bib}"
        temp_file = collector.base_path / "test_bib.tex"
        temp_file.write_text(test_content)

        deps = collector._parse_file_dependencies("test_bib.tex")
        bib_deps = [d for d in deps if d[0] in ["bibliography", "addbibresource"]]
        assert len(bib_deps) == 2
        assert ("bibliography", "refs", 1) in deps
        assert ("bibliography", "main.bib", 1) in deps  # addbibresource matches bibliography pattern