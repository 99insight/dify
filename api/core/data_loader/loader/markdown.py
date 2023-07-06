import logging
import re
from typing import Optional, List, Tuple, cast

from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.helpers import detect_file_encodings
from langchain.schema import Document

logger = logging.getLogger(__name__)


class TreeNode:
    def __init__(self, title: str, level: int, parent=None):
        self.title = title
        self.level = level
        self.parent = parent
        self.children = []
        self.content = ''

    def add_child(self, child):
        self.children.append(child)

    def add_content(self, content: str):
        self.content += content

def parse_heading(line: str):
    level = line.count('#')
    title = line[level:].strip()
    return level, title


def build_tree(markdown: str) -> TreeNode:
    root = TreeNode(title="root", level=0)
    current_node = root
    buffer = ""

    for line in markdown.splitlines():

        if line.startswith('#'):
            if buffer:
                current_node.add_content(buffer)
                buffer = ""

            level, title = parse_heading(line)

            while current_node.level >= level:
                current_node = current_node.parent

            new_node = TreeNode(title=title, level=level, parent=current_node)
            current_node.add_child(new_node)
            current_node = new_node

        else:
            buffer += line + '\n'

    if buffer:
        current_node.add_content(buffer)

    return root


def split_block(content:str):
    sp = (r'\n|。|;|!')
    content = re.split(sp,content)
    return content

def traverse_tree(node: TreeNode, depth: int = 0,length:int=0,product_info:str='',md_end_info:list=[]):
    if node.children == []:
        product_info+='#' * node.level + ' ' + node.title + '\n'
        if len(product_info+node.content)<1000:
            return product_info+node.content.replace('\t','')+'\n\n'
        else:
            contents = split_block(node.content.replace('\t',''))
            new_pro = product_info
            new_pro_cacha = ''
            for content in contents:
                if len(new_pro+content)<1000:
                    new_pro+=content+'\n'
                else:
                    new_pro_cacha += new_pro+'\n'
                    new_pro = product_info
            new_pro_cacha += new_pro
            return new_pro_cacha

    else:
        if node.title!='root':
            title = '#'*node.level+' '+node.title+'\n'
            info = node.content.replace('\n','')+'\n\n'
            product_info+=str(title+info)
        length=len(product_info)

    for child in node.children:
        info = traverse_tree(child, depth + 1,length,product_info,md_end_info)
        if child.children==[]:
            md_end_info.append(info)
        else:
            md_end_info=info

    return md_end_info



class MarkdownLoader(BaseLoader):
    """Load md files.


    Args:
        file_path: Path to the file to load.

        remove_hyperlinks: Whether to remove hyperlinks from the text.

        remove_images: Whether to remove images from the text.

        encoding: File encoding to use. If `None`, the file will be loaded
        with the default system encoding.

        autodetect_encoding: Whether to try to autodetect the file encoding
            if the specified encoding fails.
    """

    def __init__(
        self,
        file_path: str,
        remove_hyperlinks: bool = True,
        remove_images: bool = True,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = True,
    ):
        """Initialize with file path."""
        self._file_path = file_path
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images
        self._encoding = encoding
        self._autodetect_encoding = autodetect_encoding

    def load(self) -> List[Document]:
        tups = self.parse_tups(self._file_path)
        documents = []
        for header, value in tups:
            value = value.strip()
            if header is None:
                documents.append(Document(page_content=value))
            else:
                documents.append(Document(page_content=f"\n\n{header}\n{value}"))

        return documents

    def markdown_to_tups(self, markdown_text: str) -> List[Tuple[Optional[str], str]]:
        """Convert a markdown file to a dictionary.

        The keys are the headers and the values are the text under each header.

        """
        markdown_tups: List[Tuple[Optional[str], str]] = []
        mdtree = build_tree(markdown_text)
        current_text = traverse_tree(mdtree)

        # lines = markdown_text.split("\n")

        current_header = None
        # current_text = ""
        #
        # for line in lines:
        #     header_match = re.match(r"^#+\s", line)
        #     if header_match:
        #         if current_header is not None:
        #             markdown_tups.append((current_header, current_text))
        #
        #         current_header = line
        #         current_text = ""
        #     else:
        #         current_text += line + "\n"

        for line in current_text:
            header_match = re.match(r"^#+\s", line)
            if header_match:
                if current_header is not None:
                    markdown_tups.append((current_header, current_text))

                current_header = line
                current_text = ""
            else:
                current_text += line + "\n"
        markdown_tups.append((current_header, current_text))

        if current_header is not None:
            # pass linting, assert keys are defined
            markdown_tups = [
                (re.sub(r"#", "", cast(str, key)).strip(), re.sub(r"<.*?>", "", value))
                for key, value in markdown_tups
            ]
        else:
            markdown_tups = [
                (key, re.sub("\n", "", value)) for key, value in markdown_tups
            ]

        return markdown_tups

    def remove_images(self, content: str) -> str:
        """Get a dictionary of a markdown file from its path."""
        pattern = r"!{1}\[\[(.*)\]\]"
        content = re.sub(pattern, "", content)
        return content

    def remove_hyperlinks(self, content: str) -> str:
        """Get a dictionary of a markdown file from its path."""
        pattern = r"\[(.*?)\]\((.*?)\)"
        content = re.sub(pattern, r"\1", content)
        return content

    def parse_tups(self, filepath: str) -> List[Tuple[Optional[str], str]]:
        """Parse file into tuples."""
        content = ""
        try:
            with open(filepath, "r", encoding=self._encoding) as f:
                content = f.read()
        except UnicodeDecodeError as e:
            if self._autodetect_encoding:
                detected_encodings = detect_file_encodings(filepath)
                for encoding in detected_encodings:
                    logger.debug("Trying encoding: ", encoding.encoding)
                    try:
                        with open(filepath, encoding=encoding.encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {filepath}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {filepath}") from e

        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)

        if self._remove_images:
            content = self.remove_images(content)

        return self.markdown_to_tups(content)
