"""
LaTeX Project Splitter

This module provides functionality to split a LaTeX project into separate files
based on chapters and sections, using information extracted from TOC, secid, and aux files.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, List, Dict


class LatexProjectSplitter:
    """
    Split a LaTeX project into separate files based on document structure.
    
    This class takes processed TOC data (from LatexTocProcessor) and splits
    the LaTeX project into separate files, one per chapter or section.
    Filenames are derived from section titles with spaces replaced by underscores.
    
    Attributes:
        base_path (Path): Base directory path where LaTeX source files are located.
    
    Examples:
        Basic usage:
            >>> from snippets.process import LatexTocProcessor
            >>> from snippets.splitter import LatexProjectSplitter
            >>> 
            >>> processor = LatexTocProcessor("../latex_split_test_project")
            >>> data = processor.process("main.toc", "main.secid", "main.aux")
            >>> 
            >>> splitter = LatexProjectSplitter("../latex_split_test_project")
            >>> output_files = splitter.split(data, "output")
            >>> len(output_files)
            30
    """
    
    def __init__(self, base_path: Path | str):
        """
        Initialize the splitter with a base path.
        
        Args:
            base_path: Base directory path where LaTeX source files are located.
                      Can be a string or Path object. All file operations will be
                      relative to this path.
        """
        self.base_path = Path(base_path)
    
    @staticmethod
    def _sanitize_filename(title: str) -> str:
        """
        Convert a section title to a valid filename.
        
        Replaces spaces with underscores and removes invalid characters.
        
        Args:
            title: The section title to convert.
        
        Returns:
            A sanitized filename string.
        """
        # Replace spaces with underscores
        filename = title.replace(' ', '_')
        # Remove or replace invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        # Remove multiple consecutive underscores
        while '__' in filename:
            filename = filename.replace('__', '_')
        # Remove leading/trailing underscores
        filename = filename.strip('_')
        return filename
    
    def _find_file_path(self, file_path: str, relative_to: Optional[Path] = None) -> Optional[Path]:
        """
        Find the full path to a LaTeX source file.
        
        Primarily used for resolving \input commands. Tries paths relative to base_path
        first, then relative to a specified directory, then searches recursively.
        
        Args:
            file_path: The file path or filename to search for.
            relative_to: Optional directory to try relative paths from (for \input commands
                        that might be relative to the current file).
        
        Returns:
            Path to the file if found, None otherwise.
        """
        # Normalize path separators
        file_path = file_path.replace('\\', '/')
        
        # Try direct path relative to base_path (most common case)
        direct_path = self.base_path / file_path
        if direct_path.exists():
            return direct_path
        
        # Try relative to specified directory (for \input commands relative to current file)
        if relative_to and relative_to.is_dir():
            relative_path = relative_to.parent / file_path
            if relative_path.exists():
                return relative_path
        
        # Fallback: recursive search (for backward compatibility with old secid files)
        # Only search for the filename part if it looks like just a filename
        if '/' not in file_path:
            for path in self.base_path.rglob(file_path):
                if path.is_file():
                    return path
        
        return None
    
    def _extract_section_content(self, entry: Dict, all_entries: List[Dict], 
                                 current_index: int) -> str:
        """
        Extract LaTeX content for a given section/chapter entry.
        
        This method reads the source file(s) and extracts all content
        belonging to the section, including content from \input files.
        It stops at the next section of the same or higher level.
        
        **Limitation**: This algorithm assumes that each section/chapter is
        contained within a single file (where the section/chapter command is located).
        Content is extracted from that file's start line to the next section of
        same or higher level in the same file. \input commands are expanded
        recursively, but boundaries are only detected within the same file.
        
        For example:
        - If a chapter file contains sections directly (not via \\input), extracting
          the chapter will include all sections up to the next chapter.
        - If a section spans multiple files (beyond \\input), only content in the
          file where \\section is located will be extracted.
        
        Args:
            entry: Dictionary with section information (from process()).
            all_entries: List of all entries (filtered, chapters and sections only).
            current_index: Index of current entry in all_entries.
        
        Returns:
            String containing the LaTeX content for this section.
        """
        if not entry.get('file') or entry.get('line') is None:
            return f"% {entry['level']}: {entry['title']}\n% No source file information available\n"
        
        # Use the relative path directly from the entry
        file_path = self.base_path / entry['file']
        if not file_path.exists():
            # Fallback: try searching if direct path doesn't work (for backward compatibility)
            file_path = self._find_file_path(entry['file'])
            if not file_path or not file_path.exists():
                return f"% {entry['level']}: {entry['title']}\n% Source file not found: {entry['file']}\n"
        
        # Read the source file
        try:
            lines = file_path.read_text(encoding='utf-8', errors='replace').splitlines()
        except Exception as e:
            return f"% {entry['level']}: {entry['title']}\n% Error reading file: {e}\n"
        
        start_line = entry['line'] - 1  # Convert to 0-based index
        if start_line < 0 or start_line >= len(lines):
            return f"% {entry['level']}: {entry['title']}\n% Invalid line number: {entry['line']}\n"
        
        # Determine the level hierarchy
        level_hierarchy = {'chapter': 1, 'section': 2, 'subsection': 3}
        current_level = level_hierarchy.get(entry['level'], 99)
        
        # Find the end of this section (next section of same or higher level in same file)
        # Note: This algorithm assumes each section/chapter is contained in one file.
        # For sections that span multiple files, only content in the file where the
        # section command is located will be extracted. \input commands are expanded.
        end_line = len(lines)
        for i in range(current_index + 1, len(all_entries)):
            next_entry = all_entries[i]
            next_level = level_hierarchy.get(next_entry['level'], 99)
            if next_level <= current_level:
                # Found next section of same or higher level
                # Check if it's in the same file
                if next_entry.get('file') == entry['file'] and next_entry.get('line'):
                    end_line = next_entry['line'] - 1
                break
        
        # Extract content from start_line to end_line
        content_lines = lines[start_line:end_line]
        
        # Process \input commands - expand them recursively
        result_lines = []
        processed_inputs = set()  # Track processed inputs to avoid infinite recursion
        
        def expand_inputs(line: str, current_file: Optional[Path] = None, depth: int = 0) -> List[str]:
            """Recursively expand \input commands, preserving relative paths."""
            if depth > 10:  # Prevent infinite recursion
                return [line]
            
            input_match = re.search(r'\\input\{([^}]+)\}', line)
            if not input_match:
                return [line]
            
            # Get the original input path from the command (may or may not have .tex extension)
            original_input_path = input_match.group(1)
            
            # For finding the file, add .tex extension if not present
            input_file = original_input_path
            if not input_file.endswith('.tex'):
                input_file += '.tex'
            
            # Check if we've already processed this input (using the path with .tex)
            if input_file in processed_inputs:
                return [line]  # Keep original to avoid recursion
            
            # Try to find the input file (may be relative to current file or base_path)
            input_path = self._find_file_path(input_file, relative_to=current_file)
            if input_path and input_path.exists():
                try:
                    processed_inputs.add(input_file)
                    input_content = input_path.read_text(encoding='utf-8', errors='replace')
                    input_lines = input_content.splitlines()
                    
                    # Compute relative path from base_path for the comment
                    try:
                        relative_path = input_path.relative_to(self.base_path)
                        # Convert to forward slashes for LaTeX compatibility
                        relative_path_str = str(relative_path).replace('\\', '/')
                        # Remove .tex extension to match original input format
                        if relative_path_str.endswith('.tex'):
                            relative_path_str = relative_path_str[:-4]
                    except ValueError:
                        # If path is not relative to base_path, use original input path
                        relative_path_str = original_input_path
                    
                    # Recursively expand inputs in the included file
                    expanded_lines = []
                    for input_line in input_lines:
                        expanded_lines.extend(expand_inputs(input_line, current_file=input_path, depth=depth + 1))
                    
                    processed_inputs.remove(input_file)  # Allow re-inclusion in different contexts
                    return [f"% Expanded from: \\input{{{relative_path_str}}}"] + expanded_lines
                except Exception:
                    return [line]  # Keep original line if expansion fails
            else:
                return [line]  # Keep original line if file not found
        
        # Process each line, expanding inputs
        for line in content_lines:
            result_lines.extend(expand_inputs(line, current_file=file_path))
        
        return '\n'.join(result_lines) + '\n'
    
    def validate_structure(self, data: List[Dict], include_subsections: bool = False) -> List[Dict]:
        """
        Validate that the document structure satisfies the splitting algorithm's limitations.
        
        Checks if sections/chapters span multiple files, which could result in content loss.
        This is a public method that can be called independently of split().
        
        Args:
            data: List of dictionaries from LatexTocProcessor.process().
            include_subsections: If True, include subsections in validation. Default False.
        
        Returns:
            List of dictionaries with validation issue information:
            - 'entry': The entry dictionary that has the issue
            - 'index': Index of the entry in the data list
            - 'level': Section level (e.g., 'chapter', 'section')
            - 'title': Section title
            - 'issue': Description of the issue
            - 'severity': 'warning' or 'error'
            - 'affected_file': File that might be missed
            - 'affected_line': Line number in affected file
        """
        # Filter data same way as split() would
        if not include_subsections:
            filtered_data = [e for e in data if e['level'] != 'subsection']
        else:
            filtered_data = data
        
        return self._validate_structure(filtered_data)
    
    def _validate_structure(self, data: List[Dict]) -> List[Dict]:
        """
        Validate that the document structure satisfies the splitting algorithm's limitations.
        
        Checks if sections/chapters span multiple files, which could result in content loss.
        Returns a list of warnings/errors for entries that don't satisfy the limitation.
        
        Args:
            data: List of dictionaries from LatexTocProcessor.process().
        
        Returns:
            List of dictionaries with 'entry', 'level', 'title', 'issue', and 'severity' keys
            describing validation issues found.
        """
        level_hierarchy = {'chapter': 1, 'section': 2, 'subsection': 3}
        issues = []
        
        for i, entry in enumerate(data):
            current_level = level_hierarchy.get(entry['level'], 99)
            entry_file = entry.get('file')
            
            if not entry_file:
                continue
            
            # Find the next entry of same or higher level
            next_same_level_index = None
            for j in range(i + 1, len(data)):
                next_entry = data[j]
                next_level = level_hierarchy.get(next_entry['level'], 99)
                if next_level <= current_level:
                    next_same_level_index = j
                    break
            
            # Check entries between current and next same-level entry
            if next_same_level_index is not None:
                next_entry = data[next_same_level_index]
                next_entry_file = next_entry.get('file')
                
                # Check all entries between current and next same-level entry
                for k in range(i + 1, next_same_level_index):
                    intermediate_entry = data[k]
                    intermediate_level = level_hierarchy.get(intermediate_entry['level'], 99)
                    intermediate_file = intermediate_entry.get('file')
                    
                    # If there's a deeper-level entry (belongs to current section) in a different file
                    # This indicates the section spans multiple files
                    if (intermediate_level > current_level and 
                        intermediate_file and 
                        intermediate_file != entry_file):
                        issues.append({
                            'entry': entry,
                            'index': i,
                            'level': entry['level'],
                            'title': entry['title'],
                            'issue': f"Section '{entry['title']}' has nested content in different file. "
                                    f"Content from '{intermediate_file}' (line {intermediate_entry.get('line', '?')}) "
                                    f"may not be included when extracting from '{entry_file}' (line {entry.get('line', '?')}). "
                                    f"The algorithm only extracts content from the file where the section command is located.",
                            'severity': 'warning',
                            'affected_file': intermediate_file,
                            'affected_line': intermediate_entry.get('line'),
                            'source_file': entry_file,
                            'source_line': entry.get('line')
                        })
            else:
                # No next same-level entry found - check if there are any deeper entries in different files
                # that come after this entry (they would belong to this section)
                for k in range(i + 1, len(data)):
                    intermediate_entry = data[k]
                    intermediate_level = level_hierarchy.get(intermediate_entry['level'], 99)
                    intermediate_file = intermediate_entry.get('file')
                    
                    # If there's a deeper-level entry in a different file after this entry
                    if (intermediate_level > current_level and 
                        intermediate_file and 
                        intermediate_file != entry_file):
                        issues.append({
                            'entry': entry,
                            'index': i,
                            'level': entry['level'],
                            'title': entry['title'],
                            'issue': f"Section '{entry['title']}' may have content in different file. "
                                    f"Content from '{intermediate_file}' (line {intermediate_entry.get('line', '?')}) "
                                    f"comes after this section but is in a different file than '{entry_file}'. "
                                    f"This content may not be included when extracting.",
                            'severity': 'warning',
                            'affected_file': intermediate_file,
                            'affected_line': intermediate_entry.get('line'),
                            'source_file': entry_file,
                            'source_line': entry.get('line')
                        })
                    # Stop at first entry of same or higher level
                    elif intermediate_level <= current_level:
                        break
        
        return issues
    
    def split(self, data: List[Dict], output_dir: Path | str,
              include_subsections: bool = False, validate: bool = True) -> Dict[str, Path]:
        """
        Split a LaTeX project into separate files based on chapters and sections.
        
        This method takes processed TOC data and creates separate LaTeX files
        for each chapter and section. Filenames are derived from section titles
        with spaces replaced by underscores.
        
        Args:
            data: List of dictionaries from LatexTocProcessor.process().
            output_dir: Directory where output files will be created.
            include_subsections: If True, include subsections in the output. Default False.
            validate: If True (default), validate structure and warn about potential content loss.
        
        Returns:
            Dictionary mapping section titles to output file paths.
        
        Examples:
            >>> from snippets.process import LatexTocProcessor
            >>> from snippets.splitter import LatexProjectSplitter
            >>> 
            >>> processor = LatexTocProcessor("../latex_split_test_project")
            >>> data = processor.process("main.toc", "main.secid", "main.aux")
            >>> 
            >>> splitter = LatexProjectSplitter("../latex_split_test_project")
            >>> output_files = splitter.split(data, "output")
        """
        # Filter out subsections unless explicitly requested
        if not include_subsections:
            filtered_data = [e for e in data if e['level'] != 'subsection']
        else:
            filtered_data = data
        
        # Validate structure if requested (validate on filtered data that will be split)
        if validate:
            issues = self._validate_structure(filtered_data)
            if issues:
                import warnings
                for issue in issues:
                    warnings.warn(
                        f"[{issue['level']}] {issue['title']}: {issue['issue']}",
                        UserWarning,
                        stacklevel=2
                    )
                # Print summary
                print(f"\n⚠️  Validation found {len(issues)} potential issue(s) where content might be lost.")
                print("   The splitting algorithm assumes each section/chapter is in one file.")
                print("   Review the warnings above and consider restructuring your LaTeX project.\n")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store output file paths
        output_files = {}
        filename_counts = {}  # Track filename usage to handle duplicates
        
        # Process each entry
        for i, entry in enumerate(filtered_data):
            # Generate filename from title
            filename = self._sanitize_filename(entry['title'])
            if not filename:
                filename = f"{entry['level']}_{entry['number']}"
            
            # Handle duplicate filenames by adding a number suffix
            base_filename = filename
            counter = 1
            while filename in filename_counts:
                filename = f"{base_filename}_{counter}"
                counter += 1
            filename_counts[filename] = True
            
            # Add .tex extension
            filename += '.tex'
            
            # Extract content for this section
            content = self._extract_section_content(entry, filtered_data, i)
            
            # Write to output file
            output_file = output_path / filename
            output_file.write_text(content, encoding='utf-8')
            
            # Store mapping
            output_files[entry['title']] = output_file
        
        return output_files

