from process import LatexTocProcessor
from splitter import LatexProjectSplitter

if __name__ == "__main__":

    processor = LatexTocProcessor("../latex_split_test_project")
    data = processor.process("main.toc", "main.secid", "main.aux")
    splitter = LatexProjectSplitter("../latex_split_test_project")
    output_files = splitter.split(data, "output")
    print(output_files)