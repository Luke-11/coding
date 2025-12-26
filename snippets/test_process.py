"""
Comprehensive pytest tests for LatexTocProcessor class.
"""
import pytest
import tempfile
from pathlib import Path

try:
    from process import LatexTocProcessor, TocEntry, HAS_PANDAS
except ImportError:
    from snippets.process import LatexTocProcessor, TocEntry, HAS_PANDAS


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_toc_content():
    """Sample TOC file content."""
    return """\\contentsline {chapter}{[secid=1]\\numberline {1}Introduction}{1}{chapter.1}%
\\contentsline {section}{[secid=2]\\numberline {1.1}Motivation}{1}{section.1.1}%
\\contentsline {subsection}{[secid=3]\\numberline {1.1.1}Background}{1}{subsection.1.1.1}%
\\contentsline {chapter}{[secid=4]Appendix}{9}{chapter*.2}%
"""


@pytest.fixture
def sample_aux_content():
    """Sample aux file content."""
    return """\\relax 
\\newlabel{chap:intro}{{1}{1}{Introduction}{chapter.1}{}}
\\newlabel{sec:motivation}{{1.1}{1}{Motivation}{section.1.1}{}}
\\newlabel{subsec:background}{{1.1.1}{1}{Background}{subsection.1.1.1}{}}
\\newlabel{chap:appendix}{{4.1.2}{9}{Appendix}{chapter*.2}{}}
"""


@pytest.fixture
def sample_secid_content():
    """Sample secid file content."""
    return """1|document.tex|20
2|document.tex|22
3|document.tex|23
4|document.tex|78
"""


@pytest.fixture
def sample_files(temp_dir, sample_toc_content, sample_aux_content, sample_secid_content):
    """Create sample test files."""
    toc_file = temp_dir / "test.toc"
    aux_file = temp_dir / "test.aux"
    secid_file = temp_dir / "test.secid"
    
    toc_file.write_text(sample_toc_content, encoding="utf-8")
    aux_file.write_text(sample_aux_content, encoding="utf-8")
    secid_file.write_text(sample_secid_content, encoding="utf-8")
    
    return {
        "toc": toc_file,
        "aux": aux_file,
        "secid": secid_file,
        "base": temp_dir
    }


class TestTocEntry:
    """Tests for TocEntry dataclass."""
    
    def test_toc_entry_creation(self):
        """Test creating a TocEntry with all fields."""
        entry = TocEntry(
            level="chapter",
            number="1",
            title="Introduction",
            page="1",
            label_ref="chapter.1",
            label_name="chap:intro",
            secid=1,
            file="document.tex",
            line=20
        )
        assert entry.level == "chapter"
        assert entry.number == "1"
        assert entry.title == "Introduction"
        assert entry.page == "1"
        assert entry.label_ref == "chapter.1"
        assert entry.label_name == "chap:intro"
        assert entry.secid == 1
        assert entry.file == "document.tex"
        assert entry.line == 20
    
    def test_toc_entry_minimal(self):
        """Test creating a TocEntry with minimal required fields."""
        entry = TocEntry(
            level="section",
            number=None,
            title="Unnumbered Section",
            page="5"
        )
        assert entry.level == "section"
        assert entry.number is None
        assert entry.title == "Unnumbered Section"
        assert entry.page == "5"
        assert entry.label_ref is None
        assert entry.label_name is None
        assert entry.secid is None
        assert entry.file is None
        assert entry.line is None


class TestLatexTocProcessor:
    """Tests for LatexTocProcessor class."""
    
    def test_init_with_string_path(self):
        """Test initialization with string path."""
        processor = LatexTocProcessor("snippets")
        assert processor.base_path == Path("snippets")
        assert processor.entries == []
    
    def test_init_with_path_object(self):
        """Test initialization with Path object."""
        path = Path("snippets")
        processor = LatexTocProcessor(path)
        assert processor.base_path == path
        assert processor.entries == []
    
    def test_tex_unbrace(self):
        """Test _tex_unbrace static method."""
        assert LatexTocProcessor._tex_unbrace("  Hello World  ") == "Hello World"
        assert LatexTocProcessor._tex_unbrace("Text") == "Text"
        assert LatexTocProcessor._tex_unbrace("") == ""
    
    def test_find_brace_end_simple(self):
        """Test _find_brace_end with simple braces."""
        text = "{hello}"
        result = LatexTocProcessor._find_brace_end(text, 1)  # Start after '{'
        assert result == 6  # Index of closing '}'
    
    def test_find_brace_end_nested(self):
        """Test _find_brace_end with nested braces."""
        text = "{\\numberline{1}Title}"
        result = LatexTocProcessor._find_brace_end(text, 1)  # Start after opening '{'
        assert result == len(text) - 1  # Should find the closing '}'
    
    def test_find_brace_end_unbalanced(self):
        """Test _find_brace_end with unbalanced braces."""
        text = "{hello"
        result = LatexTocProcessor._find_brace_end(text, 1)
        assert result == -1
    
    def test_parse_toc_line_valid(self):
        """Test _parse_toc_line with valid input."""
        processor = LatexTocProcessor(".")
        line = "\\contentsline{chapter}{\\numberline{1}Introduction}{1}{chapter.1}"
        result = processor._parse_toc_line(line)
        assert result is not None
        level, body, page, label_ref = result
        assert level == "chapter"
        assert "\\numberline{1}Introduction" in body
        assert page == "1"
        assert label_ref == "chapter.1"
    
    def test_parse_toc_line_without_label(self):
        """Test _parse_toc_line without label argument."""
        processor = LatexTocProcessor(".")
        line = "\\contentsline{section}{Title}{5}"
        result = processor._parse_toc_line(line)
        assert result is not None
        level, body, page, label_ref = result
        assert level == "section"
        assert body == "Title"
        assert page == "5"
        assert label_ref is None
    
    def test_parse_toc_line_invalid(self):
        """Test _parse_toc_line with invalid input."""
        processor = LatexTocProcessor(".")
        assert processor._parse_toc_line("not a contentsline") is None
        assert processor._parse_toc_line("") is None
    
    def test_parse_toc(self, sample_files):
        """Test parse_toc method."""
        processor = LatexTocProcessor(sample_files["base"])
        entries = processor.parse_toc(sample_files["toc"])
        
        assert len(entries) == 4
        assert entries[0].level == "chapter"
        assert entries[0].number == "1"
        assert entries[0].title == "Introduction"
        assert entries[0].page == "1"
        assert entries[0].secid == 1
        assert entries[0].label_ref == "chapter.1"
        
        assert entries[1].level == "section"
        assert entries[1].number == "1.1"
        assert entries[1].title == "Motivation"
        assert entries[1].secid == 2
        
        assert entries[3].level == "chapter"
        assert entries[3].number is None  # Unnumbered appendix
        assert entries[3].title == "Appendix"
    
    def test_parse_toc_nonexistent_file(self, temp_dir):
        """Test parse_toc with nonexistent file."""
        processor = LatexTocProcessor(temp_dir)
        with pytest.raises(FileNotFoundError):
            processor.parse_toc("nonexistent.toc")
    
    def test_parse_aux_labels(self, sample_files):
        """Test parse_aux_labels method."""
        processor = LatexTocProcessor(sample_files["base"])
        labels = processor.parse_aux_labels(sample_files["aux"])
        
        assert len(labels) == 4
        assert labels["chapter.1"] == "chap:intro"
        assert labels["section.1.1"] == "sec:motivation"
        assert labels["subsection.1.1.1"] == "subsec:background"
        assert labels["chapter*.2"] == "chap:appendix"
    
    def test_parse_aux_labels_nonexistent_file(self, temp_dir):
        """Test parse_aux_labels with nonexistent file."""
        processor = LatexTocProcessor(temp_dir)
        with pytest.raises(FileNotFoundError):
            processor.parse_aux_labels("nonexistent.aux")
    
    def test_parse_sectpos(self, sample_files):
        """Test parse_sectpos method."""
        processor = LatexTocProcessor(sample_files["base"])
        positions = processor.parse_sectpos(sample_files["secid"])
        
        assert len(positions) == 4
        assert positions[1] == ("document.tex", 20)
        assert positions[2] == ("document.tex", 22)
        assert positions[3] == ("document.tex", 23)
        assert positions[4] == ("document.tex", 78)
    
    def test_parse_sectpos_nonexistent_file(self, temp_dir):
        """Test parse_sectpos with nonexistent file (should return empty dict)."""
        processor = LatexTocProcessor(temp_dir)
        positions = processor.parse_sectpos("nonexistent.secid")
        assert positions == {}
    
    def test_parse_sectpos_invalid_lines(self, temp_dir):
        """Test parse_sectpos with invalid lines."""
        secid_file = temp_dir / "invalid.secid"
        secid_file.write_text("invalid line\nanother invalid\n", encoding="utf-8")
        
        processor = LatexTocProcessor(temp_dir)
        positions = processor.parse_sectpos(secid_file)
        assert positions == {}
    
    def test_attach_positions(self):
        """Test _attach_positions method."""
        processor = LatexTocProcessor(".")
        entries = [
            TocEntry(level="chapter", number="1", title="Intro", page="1", secid=1),
            TocEntry(level="section", number="1.1", title="Section", page="1", secid=2),
            TocEntry(level="subsection", number="1.1.1", title="Subsection", page="1", secid=None),
        ]
        posmap = {
            1: ("document.tex", 20),
            2: ("document.tex", 22),
        }
        
        processor._attach_positions(entries, posmap)
        
        assert entries[0].file == "document.tex"
        assert entries[0].line == 20
        assert entries[1].file == "document.tex"
        assert entries[1].line == 22
        assert entries[2].file is None  # No secid, so no position attached
        assert entries[2].line is None
    
    def test_attach_labels(self):
        """Test _attach_labels method."""
        processor = LatexTocProcessor(".")
        entries = [
            TocEntry(level="chapter", number="1", title="Intro", page="1", label_ref="chapter.1"),
            TocEntry(level="section", number="1.1", title="Section", page="1", label_ref="section.1.1"),
            TocEntry(level="subsection", number="1.1.1", title="Subsection", page="1", label_ref=None),
        ]
        labelmap = {
            "chapter.1": "chap:intro",
            "section.1.1": "sec:motivation",
        }
        
        processor._attach_labels(entries, labelmap)
        
        assert entries[0].label_name == "chap:intro"
        assert entries[1].label_name == "sec:motivation"
        assert entries[2].label_name is None  # No label_ref, so no label attached
    
    def test_process(self, sample_files):
        """Test process method."""
        processor = LatexTocProcessor(sample_files["base"])
        data = processor.process()
        
        assert len(data) == 4
        assert data[0]["number"] == "1"
        assert data[0]["level"] == "chapter"
        assert data[0]["title"] == "Introduction"
        assert data[0]["page"] == "1"
        assert data[0]["file"] == "document.tex"
        assert data[0]["line"] == 20
        assert data[0]["label"] == "chap:intro"
        
        assert data[1]["number"] == "1.1"
        assert data[1]["level"] == "section"
        assert data[1]["title"] == "Motivation"
        assert data[1]["file"] == "document.tex"
        assert data[1]["line"] == 22
        assert data[1]["label"] == "sec:motivation"
        
        assert data[3]["number"] == "*"  # Unnumbered appendix
        assert data[3]["level"] == "chapter"
        assert data[3]["title"] == "Appendix"
    
    def test_process_custom_filenames(self, sample_files):
        """Test process with custom filenames."""
        processor = LatexTocProcessor(sample_files["base"])
        # Rename files
        sample_files["toc"].rename(sample_files["base"] / "custom.toc")
        sample_files["aux"].rename(sample_files["base"] / "custom.aux")
        sample_files["secid"].rename(sample_files["base"] / "custom.secid")
        
        data = processor.process("custom.toc", "custom.secid", "custom.aux")
        assert len(data) == 4
    
    def test_process_stores_entries(self, sample_files):
        """Test that process stores entries in self.entries."""
        processor = LatexTocProcessor(sample_files["base"])
        assert len(processor.entries) == 0
        
        processor.process()
        assert len(processor.entries) == 4
        assert isinstance(processor.entries[0], TocEntry)
    
    def test_to_dataframe(self, sample_files):
        """Test to_dataframe method."""
        processor = LatexTocProcessor(sample_files["base"])
        df = processor.to_dataframe()
        
        if HAS_PANDAS:
            assert df is not None
            assert len(df) == 4
            assert list(df.columns) == ["number", "level", "page", "file", "line", "label", "title"]
            assert df.iloc[0]["title"] == "Introduction"
        else:
            assert df is None
    
    def test_print_table(self, sample_files, capsys):
        """Test print_table method."""
        processor = LatexTocProcessor(sample_files["base"])
        processor.print_table()
        
        captured = capsys.readouterr()
        assert "Introduction" in captured.out
        assert "Motivation" in captured.out
        assert "number" in captured.out or "number" in captured.out.lower()
    
    def test_print_table_empty_data(self, temp_dir, capsys):
        """Test print_table with no data."""
        processor = LatexTocProcessor(temp_dir)
        # Create empty files
        (temp_dir / "empty.toc").write_text("", encoding="utf-8")
        (temp_dir / "empty.secid").write_text("", encoding="utf-8")
        (temp_dir / "empty.aux").write_text("", encoding="utf-8")
        
        processor.print_table("empty.toc", "empty.secid", "empty.aux")
        captured = capsys.readouterr()
        assert "No data to display" in captured.out or len(captured.out.strip()) == 0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_toc_without_secid(self, temp_dir):
        """Test parsing TOC without secid markers."""
        toc_content = "\\contentsline{chapter}{\\numberline{1}Title}{1}{chapter.1}%"
        toc_file = temp_dir / "no_secid.toc"
        toc_file.write_text(toc_content, encoding="utf-8")
        
        processor = LatexTocProcessor(temp_dir)
        entries = processor.parse_toc(toc_file)
        
        assert len(entries) == 1
        assert entries[0].secid is None
        assert entries[0].title == "Title"
    
    def test_toc_without_numberline(self, temp_dir):
        """Test parsing TOC without numberline command."""
        toc_content = "\\contentsline{chapter}{Title}{1}{chapter.1}%"
        toc_file = temp_dir / "no_number.toc"
        toc_file.write_text(toc_content, encoding="utf-8")
        
        processor = LatexTocProcessor(temp_dir)
        entries = processor.parse_toc(toc_file)
        
        assert len(entries) == 1
        assert entries[0].number is None
        assert entries[0].title == "Title"
    
    def test_aux_without_label_ref(self, temp_dir):
        """Test parsing aux file with labels that have no label_ref."""
        aux_content = "\\newlabel{label1}{{1}{1}{Title}{}{}}\n"
        aux_file = temp_dir / "no_ref.aux"
        aux_file.write_text(aux_content, encoding="utf-8")
        
        processor = LatexTocProcessor(temp_dir)
        labels = processor.parse_aux_labels(aux_file)
        
        assert len(labels) == 0  # Empty label_ref should be skipped
    
    def test_secid_with_extra_whitespace(self, temp_dir):
        """Test parsing secid file with extra whitespace."""
        # The regex uses ^ anchor, so leading whitespace won't match
        # Test with trailing whitespace only (which should be stripped)
        secid_content = "1|document.tex|20  \n2|document.tex|22\n"
        secid_file = temp_dir / "whitespace.secid"
        secid_file.write_text(secid_content, encoding="utf-8")
        
        processor = LatexTocProcessor(temp_dir)
        positions = processor.parse_sectpos(secid_file)
        
        assert positions[1] == ("document.tex", 20)
        assert positions[2] == ("document.tex", 22)
    
    def test_mismatched_secid(self, sample_files):
        """Test processing when secid doesn't match."""
        processor = LatexTocProcessor(sample_files["base"])
        entries = processor.parse_toc(sample_files["toc"])
        
        # Create position map with mismatched secid
        posmap = {999: ("other.tex", 100)}
        processor._attach_positions(entries, posmap)
        
        # Entries should not have file/line set
        assert entries[0].file is None
        assert entries[0].line is None
    
    def test_mismatched_label_ref(self, sample_files):
        """Test processing when label_ref doesn't match."""
        processor = LatexTocProcessor(sample_files["base"])
        entries = processor.parse_toc(sample_files["toc"])
        
        # Create label map with mismatched label_ref
        labelmap = {"nonexistent.ref": "some:label"}
        processor._attach_labels(entries, labelmap)
        
        # Entries should not have label_name set
        assert entries[0].label_name is None


class TestIntegration:
    """Integration tests using real file structure."""
    
    def test_full_workflow(self, sample_files):
        """Test complete workflow from files to output."""
        processor = LatexTocProcessor(sample_files["base"])
        
        # Parse individual files
        entries = processor.parse_toc(sample_files["toc"])
        labels = processor.parse_aux_labels(sample_files["aux"])
        positions = processor.parse_sectpos(sample_files["secid"])
        
        # Attach information
        processor._attach_positions(entries, positions)
        processor._attach_labels(entries, labels)
        
        # Verify all entries have complete information
        assert len(entries) == 4
        assert all(e.file is not None for e in entries)
        assert all(e.line is not None for e in entries)
        assert all(e.label_name is not None for e in entries)
        
        # Process and verify output
        data = processor.process()
        assert len(data) == 4
        assert all("file" in d for d in data)
        assert all("line" in d for d in data)
        assert all("label" in d for d in data)

