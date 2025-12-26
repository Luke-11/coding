from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set

try:
    import pandas as pd
    HAS_PANDAS = True
except (ImportError, ValueError, Exception):
    HAS_PANDAS = False

from dependency_collector import DependencyEntry, LatexDependencyCollector

@dataclass
class TocEntry:
    """
    Represents a single entry from a LaTeX table of contents.
    
    This dataclass stores all information extracted from TOC, secid, and aux files
    for a single document section (chapter, section, subsection, etc.).
    
    Attributes:
        level: The sectioning level (e.g., "chapter", "section", "subsection").
        number: The section number as a string (e.g., "1", "1.1", "1.1.1").
               None if the section is unnumbered (e.g., starred sections).
        title: The title text of the section.
        page: The page number where the section appears (as a string).
        label_ref: The internal LaTeX label reference from the TOC file
                  (e.g., "chapter.1", "section.1.1"). This is used to match
                  with labels in the aux file. None if not present.
        label_name: The actual label name defined in the LaTeX source
                   (e.g., "chap:intro", "sec:motivation"). Extracted from
                   the aux file. None if not found or not present.
        secid: The section ID number from the secid package instrumentation.
              Used to match with position information in the secid file.
              None if not present.
        file: The source filename where this section is defined.
             Extracted from the secid file. None if not found.
        line: The line number in the source file where this section is defined.
             Extracted from the secid file. None if not found.
    
    Examples:
        >>> entry = TocEntry(
        ...     level="chapter",
        ...     number="1",
        ...     title="Introduction",
        ...     page="1",
        ...     label_ref="chapter.1",
        ...     label_name="chap:intro",
        ...     secid=1,
        ...     file="document.tex",
        ...     line=20
        ... )
    """
    level: str
    number: Optional[str]
    title: str
    page: Optional[str]
    label_ref: Optional[str] = None  # The label reference from TOC (e.g., "chapter.1")
    label_name: Optional[str] = None  # The actual label name from aux (e.g., "chap:intro")
    secid: Optional[int] = None
    file: Optional[str] = None
    line: Optional[int] = None

class LatexTocProcessor:
    """
    Process LaTeX TOC, secid, and aux files to extract structured document information.
    
    This class parses LaTeX auxiliary files generated during document compilation
    to extract comprehensive information about document structure, including:
    - Section hierarchy and numbering from TOC files
    - Source file locations from secid files
    - Label definitions from aux files
    
    The processor handles nested braces in LaTeX commands and matches information
    across multiple files using section IDs and label references.
    
    Attributes:
        base_path (Path): Base directory path where input files are located.
        entries (List[TocEntry]): List of parsed TOC entries. Populated after
                                  calling process() or parse_toc().
        NUMBERLINE_RE: Compiled regex pattern for matching \\numberline commands.
        SECID_RE: Compiled regex pattern for matching [secid=N] markers.
        AUX_NEWLABEL_RE: Compiled regex pattern for matching \\newlabel commands.
        SECTPOS_RE: Compiled regex pattern for matching secid file entries.
    
    Examples:
        Basic usage:
            >>> processor = LatexTocProcessor("snippets")
            >>> processor.print_table()
        
        Get DataFrame:
            >>> processor = LatexTocProcessor("snippets")
            >>> df = processor.to_dataframe("document.toc", "document.secid", "document.aux")
        
        Get raw data:
            >>> processor = LatexTocProcessor("snippets")
            >>> data = processor.process()
            >>> for entry in data:
            ...     print(entry['title'])
    """
    
    # Regex patterns for parsing
    NUMBERLINE_RE = re.compile(r"""\\numberline\s*\{(?P<num>[^}]+)\}(?P<title>.*)""", re.DOTALL)
    SECID_RE = re.compile(r"""\[secid=(?P<secid>\d+)\]""")
    AUX_NEWLABEL_RE = re.compile(
        r"""\\newlabel\{(?P<label_name>[^}]+)\}\{\{(?P<num>[^}]*)\}\{(?P<page>[^}]*)\}\{(?P<title>[^}]*)\}\{(?P<label_ref>[^}]*)\}""",
        re.DOTALL,
    )
    SECTPOS_RE = re.compile(r"""^(?P<secid>\d+)\|(?P<file>[^|]+)\|(?P<line>\d+)\s*$""")
    
    def __init__(self, base_path: Path | str):
        """
        Initialize the processor with a base path.
        
        Args:
            base_path: Base directory path where TOC, secid, and aux files are located.
                      Can be a string or Path object. All file operations will be
                      relative to this path.
        
        Raises:
            TypeError: If base_path cannot be converted to a Path.
        
        Examples:
            >>> processor = LatexTocProcessor("/path/to/latex/files")
            >>> processor = LatexTocProcessor(Path("snippets"))
        """
        self.base_path = Path(base_path)
        self.entries: List[TocEntry] = []
    
    @staticmethod
    def _tex_unbrace(s: str) -> str:
        """
        Remove braces and clean up LaTeX text.
        
        This is a lightweight text cleaning function that strips whitespace.
        For more complex LaTeX command removal, a dedicated LaTeX parser would
        be needed.
        
        Args:
            s: Input string that may contain LaTeX formatting.
        
        Returns:
            Cleaned string with leading/trailing whitespace removed.
        
        Examples:
            >>> LatexTocProcessor._tex_unbrace("  Introduction  ")
            'Introduction'
        """
        return s.strip()
    
    @staticmethod
    def _find_brace_end(text: str, start: int) -> int:
        """
        Find the matching closing brace, handling nested braces.
        
        This method implements a brace-matching algorithm that correctly handles
        nested braces by maintaining a depth counter. It starts from a given
        position (assumed to be just after an opening brace) and finds the
        corresponding closing brace.
        
        Args:
            text: The string to search in.
            start: The starting position (should be the character after '{').
        
        Returns:
            The index of the matching closing brace, or -1 if no matching
            brace is found (unbalanced braces).
        
        Examples:
            >>> processor = LatexTocProcessor(".")
            >>> text = "\\contentsline{chapter}{Title}{1}"
            >>> end = processor._find_brace_end(text, 17)  # After '{' of "chapter"
            >>> text[16:end+1]  # Extract the matched content
            'chapter'
        """
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        return i - 1 if depth == 0 else -1
    
    def _parse_toc_line(self, line: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
        r"""
        Parse a single \\contentsline entry, handling nested braces in body.
        
        This method parses a LaTeX \\contentsline command which has the format:
        \\contentsline{level}{body}{page}{label}
        
        The method uses brace matching to correctly handle nested braces in the
        body argument (e.g., \numberline{1.1}Title with nested braces).
        
        Args:
            line: A single line from the TOC file containing a \\contentsline command.
        
        Returns:
            A tuple containing (level, body, page, label_ref) if parsing succeeds,
            or None if the line doesn't contain a valid \\contentsline command.
            - level: Section level (e.g., "chapter", "section")
            - body: The body text, may contain \numberline and [secid=N] markers
            - page: Page number as a string
            - label_ref: Label reference (e.g., "chapter.1") or None if not present
        
        Examples:
            >>> processor = LatexTocProcessor(".")
            >>> line = "\\contentsline{chapter}{\\numberline {1}Introduction}{1}{chapter.1}"
            >>> result = processor._parse_toc_line(line)
            >>> result[0]  # level
            'chapter'
        """
        # Match \contentsline {level}{body}{page}{label}
        # We need to find the boundaries by counting braces
        if not line.strip().startswith('\\contentsline'):
            return None
        
        # Find the opening brace after \contentsline
        start = line.find('{')
        if start == -1:
            return None
        
        # Find level (first argument)
        level_end = self._find_brace_end(line, start + 1)
        if level_end == -1:
            return None
        level = line[start + 1:level_end]
        
        # Find body (second argument) - starts after level's closing brace
        body_start = line.find('{', level_end + 1)
        if body_start == -1:
            return None
        body_end = self._find_brace_end(line, body_start + 1)
        if body_end == -1:
            return None
        body = line[body_start + 1:body_end]
        
        # Find page (third argument)
        page_start = line.find('{', body_end + 1)
        if page_start == -1:
            return None
        page_end = self._find_brace_end(line, page_start + 1)
        if page_end == -1:
            return None
        page = line[page_start + 1:page_end]
        
        # Find label (fourth argument, optional)
        label_ref = None
        label_start = line.find('{', page_end + 1)
        if label_start != -1:
            label_end = self._find_brace_end(line, label_start + 1)
            if label_end != -1:
                label_ref = line[label_start + 1:label_end]
        
        return (level, body, page, label_ref)
    
    def parse_toc(self, toc_path: Path | str) -> List[TocEntry]:
        """
        Parse TOC file and return list of TocEntry objects.
        
        This method reads a LaTeX .toc file and extracts all \\contentsline
        entries. It processes each entry to extract:
        - Section level and numbering
        - Section titles
        - Page numbers
        - Section IDs (if present from secid package)
        - Label references
        
        The method handles LaTeX commands in titles (like \\numberline) and
        extracts section IDs from [secid=N] markers added by the secid package.
        
        Args:
            toc_path: Path to the .toc file. Can be a string or Path object.
                     Can be absolute or relative to base_path.
        
        Returns:
            A list of TocEntry objects, one for each section entry in the TOC.
            Empty list if the file doesn't exist or contains no valid entries.
        
        Raises:
            FileNotFoundError: If the TOC file doesn't exist.
            UnicodeDecodeError: If the file cannot be read as UTF-8 (handled
                              with errors="replace").
        
        Examples:
            >>> processor = LatexTocProcessor("snippets")
            >>> entries = processor.parse_toc("test.toc")
            >>> len(entries)
            28
            >>> entries[0].title
            'Introduction'
        """
        toc_path = Path(toc_path)
        text = toc_path.read_text(encoding="utf-8", errors="replace")
        entries: List[TocEntry] = []
        for line in text.splitlines():
            result = self._parse_toc_line(line)
            if result is None:
                continue
            level, body, page, label_ref = result
            
            level = level.strip()
            body = body.strip()
            page = page.strip()
            label_ref = label_ref.strip() if label_ref else None

            # Extract secid if present
            secid = None
            secid_match = self.SECID_RE.search(body)
            if secid_match:
                secid = int(secid_match.group("secid"))
                # Remove secid marker from body
                body = self.SECID_RE.sub("", body).strip()

            num = None
            title = body
            mn = self.NUMBERLINE_RE.match(body)
            if mn:
                num = mn.group("num").strip()
                title = mn.group("title").strip()

            # very light cleanup
            title = self._tex_unbrace(title)

            entries.append(TocEntry(level=level, number=num, title=title, page=page, label_ref=label_ref, secid=secid))
        return entries

    def parse_aux_labels(self, aux_path: Path | str) -> Dict[str, str]:
        """
        Parse aux file and return a mapping from label_ref to label_name.
        
        This method extracts label definitions from a LaTeX .aux file. The aux
        file contains \\\\newlabel commands that map internal LaTeX label references
        (like "chapter.1") to user-defined label names (like "chap:intro").
        
        The format of \\newlabel in aux files is:
        \\newlabel{label_name}{{num}{page}{title}{label_ref}{}}
        
        Args:
            aux_path: Path to the .aux file. Can be a string or Path object.
                     Can be absolute or relative to base_path.
        
        Returns:
            A dictionary mapping label_ref (e.g., "chapter.1") to label_name
            (e.g., "chap:intro"). Empty dictionary if file doesn't exist or
            contains no valid labels.
        
        Raises:
            FileNotFoundError: If the aux file doesn't exist.
            UnicodeDecodeError: If the file cannot be read as UTF-8 (handled
                              with errors="replace").
        
        Examples:
            >>> processor = LatexTocProcessor("snippets")
            >>> labels = processor.parse_aux_labels("test.aux")
            >>> labels.get("chapter.1")
            'chap:intro'
        """
        aux_path = Path(aux_path)
        text = aux_path.read_text(encoding="utf-8", errors="replace")
        out: Dict[str, str] = {}  # label_ref -> label_name
        for m in self.AUX_NEWLABEL_RE.finditer(text):
            label_name = m.group("label_name").strip()
            label_ref = m.group("label_ref").strip()
            if label_ref:
                out[label_ref] = label_name
        return out

    def parse_sectpos(self, sectpos_path: Path | str) -> Dict[int, Tuple[str, int]]:
        """
        Parse secid file and return mapping from secid to (file, line).
        
        This method reads a .secid file generated by the secid LaTeX package.
        The file contains lines in the format:
        <secid>|<file>|<line>
        
        Each line maps a section ID number to its source file location.
        
        Args:
            sectpos_path: Path to the .secid file. Can be a string or Path object.
                         Can be absolute or relative to base_path.
        
        Returns:
            A dictionary mapping secid (integer) to a tuple of (file_path, line_number).
            Empty dictionary if the file doesn't exist or contains no valid entries.
            The file_path is stored as-is from the secid file (typically a relative path
            like "chapters/ch01/ch01.tex" when using the updated secid.sty).
        
        Raises:
            UnicodeDecodeError: If the file cannot be read as UTF-8 (handled
                              with errors="replace").
        
        Note:
            This method does not raise FileNotFoundError if the file doesn't exist.
            It simply returns an empty dictionary, allowing the secid file to be optional.
        
        Examples:
            >>> processor = LatexTocProcessor("snippets")
            >>> positions = processor.parse_sectpos("test.secid")
            >>> positions.get(1)
            ('test.tex', 20)
        """
        sectpos_path = Path(sectpos_path)
        out: Dict[int, Tuple[str, int]] = {}
        if not sectpos_path.exists():
            return out
        for line in sectpos_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = self.SECTPOS_RE.match(line)
            if not m:
                continue
            secid = int(m.group("secid"))
            out[secid] = (m.group("file"), int(m.group("line")))
        return out

    def _attach_positions(self, entries: List[TocEntry], posmap: Dict[int, Tuple[str, int]]) -> None:
        """
        Match entries by their secid number to the position map.
        
        This internal method enriches TocEntry objects with source file location
        information by matching their secid values with entries in the position map.
        The file and line attributes of matching entries are updated in-place.
        
        Args:
            entries: List of TocEntry objects to enrich. Modified in-place.
            posmap: Dictionary mapping secid (int) to (filename, line_number) tuples.
        
        Returns:
            None. The entries list is modified in-place.
        
        Examples:
            >>> processor = LatexTocProcessor(".")
            >>> entries = [TocEntry(level="chapter", number="1", title="Intro", page="1", secid=1)]
            >>> posmap = {1: ("document.tex", 20)}
            >>> processor._attach_positions(entries, posmap)
            >>> entries[0].file
            'document.tex'
            >>> entries[0].line
            20
        """
        for e in entries:
            if e.secid is not None and e.secid in posmap:
                e.file, e.line = posmap[e.secid]

    def _attach_labels(self, entries: List[TocEntry], labelmap: Dict[str, str]) -> None:
        """
        Match entries by their label_ref to get the label_name from aux file.
        
        This internal method enriches TocEntry objects with user-defined label names
        by matching their label_ref values with entries in the label map.
        The label_name attribute of matching entries is updated in-place.
        
        Args:
            entries: List of TocEntry objects to enrich. Modified in-place.
            labelmap: Dictionary mapping label_ref (e.g., "chapter.1") to
                     label_name (e.g., "chap:intro").
        
        Returns:
            None. The entries list is modified in-place.
        
        Examples:
            >>> processor = LatexTocProcessor(".")
            >>> entries = [TocEntry(level="chapter", number="1", title="Intro", page="1", label_ref="chapter.1")]
            >>> labelmap = {"chapter.1": "chap:intro"}
            >>> processor._attach_labels(entries, labelmap)
            >>> entries[0].label_name
            'chap:intro'
        """
        for e in entries:
            if e.label_ref and e.label_ref in labelmap:
                e.label_name = labelmap[e.label_ref]

    def process(self, toc_filename: str = "test.toc", secid_filename: str = "test.secid", aux_filename: str = "test.aux") -> List[Dict]:
        """
        Process all files and return structured data.
        
        This is the main processing method that orchestrates parsing of all
        input files (TOC, secid, aux) and combines the information into a
        unified data structure. It:
        1. Parses the TOC file to extract section entries
        2. Parses the secid file to get source file locations
        3. Parses the aux file to get label definitions
        4. Matches and enriches entries with position and label information
        5. Converts TocEntry objects to dictionaries for easy consumption
        
        Args:
            toc_filename: Name of the TOC file (relative to base_path).
                         Defaults to "test.toc".
            secid_filename: Name of the secid file (relative to base_path).
                           Defaults to "test.secid".
            aux_filename: Name of the aux file (relative to base_path).
                         Defaults to "test.aux".
        
        Returns:
            A list of dictionaries, each representing a section entry with keys:
            - 'number': Section number (str) or '*' for unnumbered sections
            - 'level': Section level (str, e.g., "chapter", "section")
            - 'page': Page number (str)
            - 'file': Source file path relative to base_path (str) or None if not found.
                     Example: "chapters/ch01/ch01.tex" or "chapters/ch02.tex"
            - 'line': Line number (int) or None if not found
            - 'label': Label name (str) or None if not found
            - 'title': Section title (str)
        
        Note:
            This method also populates self.entries with TocEntry objects,
            which can be accessed directly if needed.
        
        Examples:
            >>> processor = LatexTocProcessor("snippets")
            >>> data = processor.process()
            >>> len(data)
            28
            >>> data[0]['title']
            'Introduction'
            >>> data[0]['label']
            'chap:intro'
        """
        # Parse all files
        self.entries = self.parse_toc(self.base_path / toc_filename)
        posmap = self.parse_sectpos(self.base_path / secid_filename)
        labelmap = self.parse_aux_labels(self.base_path / aux_filename)
        
        # Attach positions and labels
        self._attach_positions(self.entries, posmap)
        self._attach_labels(self.entries, labelmap)

        # Convert to list of dicts
        data = []
        for e in self.entries:
            # Preserve relative path if available, otherwise use filename
            file_path = None
            if e.file:
                # Normalize path separators to forward slashes for consistency
                file_path = e.file.replace('\\', '/')
                # If it's an absolute path, try to make it relative to base_path
                try:
                    abs_path = Path(file_path)
                    if abs_path.is_absolute():
                        try:
                            file_path = str(abs_path.relative_to(self.base_path)).replace('\\', '/')
                        except ValueError:
                            # If not relative to base_path, keep as-is
                            pass
                except Exception:
                    # If path parsing fails, keep original
                    pass
            
            data.append({
                'number': e.number if e.number else '*',
                'level': e.level,
                'page': e.page,
                'file': file_path,
                'line': e.line,
                'label': e.label_name,
                'title': e.title
            })
        return data
    
    def process_filtered(self, toc_filename: str = "test.toc", secid_filename: str = "test.secid", 
                        aux_filename: str = "test.aux", exclude_levels: Optional[List[str]] = None) -> List[Dict]:
        """
        Process all files and return structured data, optionally filtering out certain levels.
        
        This method is similar to process(), but allows filtering out specific section levels
        (e.g., subsections) from the returned table.
        
        Args:
            toc_filename: Name of the TOC file (relative to base_path). Defaults to "test.toc".
            secid_filename: Name of the secid file (relative to base_path). Defaults to "test.secid".
            aux_filename: Name of the aux file (relative to base_path). Defaults to "test.aux".
            exclude_levels: List of level names to exclude (e.g., ['subsection']). 
                          If None, defaults to ['subsection'].
        
        Returns:
            A filtered list of dictionaries, each representing a section entry.
        
        Examples:
            >>> processor = LatexTocProcessor("snippets")
            >>> # Get only chapters and sections (exclude subsections)
            >>> data = processor.process_filtered()
            >>> # Get only chapters
            >>> data = processor.process_filtered(exclude_levels=['section', 'subsection'])
        """
        if exclude_levels is None:
            exclude_levels = ['subsection']
        
        all_data = self.process(toc_filename, secid_filename, aux_filename)
        filtered_data = [e for e in all_data if e['level'] not in exclude_levels]
        return filtered_data
    
    def split_project(self, output_dir: Path | str,
                      toc_filename: str = "main.toc",
                      secid_filename: str = "main.secid",
                      aux_filename: str = "main.aux",
                      include_subsections: bool = False) -> Dict[str, Path]:
        """
        Convenience method to split a LaTeX project using LatexProjectSplitter.
        
        This method processes the TOC files and then uses LatexProjectSplitter
        to create separate files for each chapter and section.
        
        Args:
            output_dir: Directory where output files will be created.
            toc_filename: Name of the TOC file (relative to base_path).
            secid_filename: Name of the secid file (relative to base_path).
            aux_filename: Name of the aux file (relative to base_path).
            include_subsections: If True, include subsections in the output. Default False.
        
        Returns:
            Dictionary mapping section titles to output file paths.
        
        Examples:
            >>> processor = LatexTocProcessor("../latex_split_test_project")
            >>> output_files = processor.split_project("output")
        """
        # Import here to avoid circular imports
        from snippets.splitter import LatexProjectSplitter
        
        # Process files to get the data
        data = self.process(toc_filename, secid_filename, aux_filename)
        
        # Use the splitter
        splitter = LatexProjectSplitter(self.base_path)
        return splitter.split(data, output_dir, include_subsections=include_subsections)
    
    def to_dataframe(self, toc_filename: str = "test.toc", secid_filename: str = "test.secid", aux_filename: str = "test.aux"):
        """
        Process files and return a pandas DataFrame.
        
        This method processes all input files and returns the results as a
        pandas DataFrame, which provides convenient data manipulation and
        analysis capabilities.
        
        Args:
            toc_filename: Name of the TOC file (relative to base_path).
                         Defaults to "test.toc".
            secid_filename: Name of the secid file (relative to base_path).
                           Defaults to "test.secid".
            aux_filename: Name of the aux file (relative to base_path).
                        Defaults to "test.aux".
        
        Returns:
            A pandas DataFrame with columns: number, level, page, file, line,
            label, title. Returns None if pandas is not available or cannot
            be imported.
        
        Note:
            This method requires pandas to be installed. If pandas is not
            available, use process() to get a list of dictionaries instead.
        
        Examples:
            >>> processor = LatexTocProcessor("snippets")
            >>> df = processor.to_dataframe()
            >>> df.head()
            >>> df[df['level'] == 'chapter']
            >>> df.to_csv('toc_export.csv')
        """
        if not HAS_PANDAS:
            return None
        data = self.process(toc_filename, secid_filename, aux_filename)
        return pd.DataFrame(data)
    
    def print_table(self, toc_filename: str = "test.toc", secid_filename: str = "test.secid", aux_filename: str = "test.aux"):
        """
        Process files and print a formatted table to stdout.
        
        This method processes all input files and displays the results as a
        nicely formatted table. It automatically uses pandas DataFrame formatting
        if available, otherwise falls back to a custom table formatter.
        
        Args:
            toc_filename: Name of the TOC file (relative to base_path).
                         Defaults to "test.toc".
            secid_filename: Name of the secid file (relative to base_path).
                           Defaults to "test.secid".
            aux_filename: Name of the aux file (relative to base_path).
                         Defaults to "test.aux".
        
        Returns:
            None. Output is printed to stdout.
        
        Examples:
            >>> processor = LatexTocProcessor("snippets")
            >>> processor.print_table()
            number | level      | page | file     | line | label                    | title
            ...
        """
        data = self.process(toc_filename, secid_filename, aux_filename)
        
        if HAS_PANDAS:
            df = pd.DataFrame(data)
            print(df.to_string(index=False))
        else:
            # Fallback: print as formatted table
            if not data:
                print("No data to display")
            else:
                # Print header
                headers = ['number', 'level', 'page', 'file', 'line', 'label', 'title']
                col_widths = {h: max(len(str(row.get(h, ''))) for row in data + [dict(zip(headers, headers))]) 
                             for h in headers}
                col_widths = {h: max(col_widths[h], len(h)) for h in headers}
                
                # Print header
                header_row = ' | '.join(f"{h:<{col_widths[h]}}" for h in headers)
                print(header_row)
                print('-' * len(header_row))
                
                # Print data rows
                for row in data:
                    print(' | '.join(f"{str(row.get(h, '')):<{col_widths[h]}}" for h in headers))

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    show_graph = "--graph" in sys.argv
    show_graph_info = "--graph-info" in sys.argv
    save_graph = "--save-graph" in sys.argv
    
    processor = LatexTocProcessor(".")
    processor.print_table()
    print("--------------------------------")
    processor=LatexTocProcessor("../latex_split_test_project")
    toc_file = "main.toc"
    secid_file = "main.secid"
    aux_file = "main.aux"
    processor.print_table(toc_filename=toc_file, secid_filename=secid_file, aux_filename=aux_file  )
    
    print("\n" + "="*50)
    print("DEPENDENCY COLLECTION TEST")
    print("="*50)
    
    collector = LatexDependencyCollector("../latex_split_test_project")
    collector.print_table("main.tex")
    
    if show_graph_info:
        print("\n" + "="*50)
        print("DEPENDENCY GRAPH INFO")
        print("="*50)
        collector.print_graph_info("main.tex")
    
    if show_graph or save_graph:
        print("\n" + "="*50)
        print("DEPENDENCY GRAPH VISUALIZATION")
        print("="*50)
        
        fig = collector.visualize_dependency_graph("main.tex")
        if fig:
            if save_graph:
                success = collector.save_dependency_graph_separate(
                    "main.tex",
                    "dependency_graph.html",
                    "dependency_graph.json"
                )
                if success:
                    print("Graph saved as separate files: 'dependency_graph.html' and 'dependency_graph.json'")
                else:
                    print("Failed to save graph files")
            else:
                print("Graph created. Use --save-graph to save as separate HTML and JSON files.")
                print("In a Jupyter notebook, you can display with: fig.show()")
        else:
            print("Failed to create graph visualization")
    # tree= collector.get_dependency_tree("main.tex")
    # for dep in tree:
    #     print(dep)
