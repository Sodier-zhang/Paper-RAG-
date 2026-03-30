import os
import re
import json

from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger


class DocumentCleaner:
    def __init__(self, documents_dir: str):
        self.input_path = Path(documents_dir)

        if self.input_path.stem.endswith("_documents"):
            self.documents_dir = self.input_path
        elif self.input_path.is_dir():
            doc_dirs = list(self.input_path.glob("*_documents"))
            if doc_dirs:
                self.documents_dir = doc_dirs[0]
            else:
                self.documents_dir = self.input_path
        else:
            self.documents_dir = self.input_path

        self.output_dir = Path(r"D:\AI\paper_RAG\parse_pdf\parsed_pdf")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_documents(self) -> List[Dict]:
        documents = []
        json_files = sorted(
            self.documents_dir.glob("page_*.json"),
            key=lambda x: int(x.stem.split("_")[1])
        )

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
                documents.append(doc_data)

        logger.info(f"加载了 {len(documents)} 个文档页面")
        return documents

    def _is_header_footer(self, line: str) -> bool:
        if not line:
            return False

        line_lower = line.lower()

        if re.match(r'^\d+\s+[A-Z][a-z]+\s+et\s+al\.?$', line):
            return True
        if re.match(r'^[A-Z].*\s+et\s+al\.\s+\d+$', line):
            return True
        if re.match(r'^#\s*\\d+\s+\S.*et\s+al', line, re.IGNORECASE):
            return True
        if re.match(r'^#\s+[A-Z].*\s+et\s+al', line, re.IGNORECASE):
            return True
        if re.match(r'^#\s*\\d+\s*$', line):
            return True
        if re.match(r'^#.*\\[\d+\\]', line):
            return True
        if re.match(r'^[A-Z][A-Za-z\s]+for\s+[A-Z]', line) and len(line) < 60:
            return True
        if re.match(r'^\d+$', line):
            return True
        if re.match(r'^Page\s+\d+$', line, re.IGNORECASE):
            return True
        if re.match(r'^\d+\s*-\s*\d+$', line):
            return True
        if len(line) < 15 and any(c.isdigit() for c in line):
            return True

        if re.match(r'^#\s+[A-Z]\.\s*[A-Z][a-z]+\s+et\s+al\.?\s*$', line):
            return True

        if re.match(r'^#\s+[A-Z][a-z]+.*\s+for\s+[A-Z][a-z]+.*\s+[A-Z][a-z]+\s*$', line):
            return True

        header_keywords = {"copyright", "doi", "article", "conference", "journal", "accepted", "published"}
        if any(kw in line_lower for kw in header_keywords) and len(line) < 80:
            return True

        return False

    def _is_author_info(self, line: str) -> bool:
        if not line:
            return False

        line_stripped = line.strip()

        if re.match(r'^#\s*\d+\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+\s*,?\s*(et\s+al)?\.?\s*$', line_stripped):
            return True
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s*\d+.*$', line_stripped):
            return True
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s*$', line_stripped):
            return True
        if re.match(r'^\d+\s+School.*University', line_stripped):
            return True
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+,\s*[A-Z][a-z]+\s+[A-Z][a-z]+', line_stripped):
            return True
        if re.match(r'.*@.*\.(com|cn|edu|org)$', line_stripped):
            return True
        if re.match(r'^asyalanwu@.*', line_stripped):
            return True
        if re.match(r'^asjgwucn@.*', line_stripped):
            return True

        if re.match(r'^#\s*\d+\s+[A-Z][a-z]+\s+[A-Z][a-z]+.*?et\s+al', line_stripped, re.IGNORECASE):
            return True
        if re.match(r'^#\s*[A-Z]\.\s*[A-Z][a-z]+\s+et\s+al', line_stripped):
            return True
        if re.match(r'^#\s*\d+\s+[A-Z]\.\s*[A-Z][a-z]+\s+et\s+al', line_stripped):
            return True
        if re.match(r'^#\s+\d+\s+[A-Z]\.\s*[A-Z][a-z]+\s+et\s+al', line_stripped):
            return True
        if re.match(r'^#\s*\d+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+et\s+al', line_stripped):
            return True

        if re.match(r'^#\s*\d+\s+[A-Z]\.\s+[A-Z][a-z]+\s+et\s+al', line_stripped):
            return True

        if re.match(r'^#\s*\d+\s+[A-Z]\.\s+[A-Z][a-z]+\s+et\s+al\.?', line_stripped):
            return True

        if re.match(r'^#\s+\d+\s+[A-Z]\.\s*[A-Za-z]+\s+et\s+al\.?', line_stripped):
            return True

        if re.match(r'^#\s+[A-Z]\.\s+[A-Z][a-z]+\s+et\s+al', line_stripped):
            return True

        if re.match(r'^#\s+\d+\s+School\s+of', line_stripped):
            return True
        if re.match(r'^#\s+\d+\s+Department\s+of', line_stripped):
            return True
        if re.match(r'^#\s+\d+\s+University', line_stripped):
            return True

        if re.match(r'^#\s*[A-Z][a-z]+@[a-z]+\.(com|cn|edu|org)', line_stripped):
            return True

        if re.match(r'^#\s*$', line_stripped):
            return True

        if re.match(r'^#\s*\d+\s+[A-Z][a-z]+\s+et\s+al', line_stripped, re.IGNORECASE):
            return True

        return False

    def _convert_to_latex(self, text: str) -> str:
        unicode_to_latex = {
            '∈': r'\in', '∑': r'\sum', '∏': r'\prod', '∪': r'\cup', '∩': r'\cap',
            '⊆': r'\subseteq', '⊇': r'\supseteq', '∅': r'\emptyset', '∞': r'\infty',
            '∂': r'\partial', '∇': r'\nabla', '√': r'\sqrt', '≤': r'\leq', '≥': r'\geq',
            '≠': r'\neq', '≈': r'\approx', '≡': r'\equiv', '∝': r'\propto', '°': r'^\circ',
            '·': r'\cdot', '×': r'\times', '÷': r'\div', '±': r'\pm', '∓': r'\mp',
            'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta', 'ε': r'\epsilon',
            'ζ': r'\zeta', 'η': r'\eta', 'θ': r'\theta', 'ι': r'\iota', 'κ': r'\kappa',
            'λ': r'\lambda', 'μ': r'\mu', 'ν': r'\nu', 'ξ': r'\xi', 'π': r'\pi',
            'ρ': r'\rho', 'σ': r'\sigma', 'τ': r'\tau', 'υ': r'\upsilon', 'φ': r'\phi',
            'χ': r'\chi', 'ψ': r'\psi', 'ω': r'\omega', 'Γ': r'\Gamma', 'Δ': r'\Delta',
            'Θ': r'\Theta', 'Λ': r'\Lambda', 'Ξ': r'\Xi', 'Π': r'\Pi', 'Σ': r'\Sigma',
            'Υ': r'\Upsilon', 'Φ': r'\Phi', 'Ψ': r'\Psi', 'Ω': r'\Omega', '∀': r'\forall',
            '∃': r'\exists', '∧': r'\land', '∨': r'\lor', '¬': r'\neg', '⊕': r'\oplus',
            '⊗': r'\otimes', '⊖': r'\ominus', '⋃': r'\bigcup', '⋂': r'\bigcap',
            '⋁': r'\bigvee', '⋀': r'\bigwedge', '∘': r'\circ', '•': r'\bullet',
            '′': r"^{\prime}", '″': r"^{\prime\prime}", '‴': r"^{\prime\prime\prime}",
            '←': r'\leftarrow', '↑': r'\uparrow', '→': r'\rightarrow', '↓': r'\downarrow',
            '↔': r'\leftrightarrow', '⇔': r'\Leftrightarrow', '⇒': r'\Rightarrow',
            '⇐': r'\Leftarrow', '↦': r'\mapsto', '· · ·': r'\cdots', '...': r'\dots',
            '−': r'-', '–': r'-', '—': r'---'
        }

        result = text
        for unicode_char, latex_char in unicode_to_latex.items():
            result = result.replace(unicode_char, latex_char)

        result = re.sub(r'(\d+)\s*\^\s*(\d+)', r'\1^{\2}', result)

        result = re.sub(r'\\sum([a-zA-Z])=([\dN])\{([^\}]+)\}', r'\\sum_{\1=\2^{\3}}', result)
        result = re.sub(r'\\sum([a-zA-Z])=([\dN])', r'\\sum_{\1=\2}', result)
        result = re.sub(r'\\sum([a-zA-Z])([a-zA-Z])', r'\\sum_{\1\2}', result)
        result = re.sub(r'\\sum([a-zA-Z])', r'\\sum_{\1}', result)

        result = re.sub(r'\\(theta|alpha|beta|gamma|delta|epsilon|zeta|eta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)_([a-zA-Z0-9]{1,3})', r'\\\1_{\2}', result)

        result = re.sub(r'\\(theta|alpha|beta|gamma|delta|epsilon|zeta|eta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)([a-zA-Z0-9]{1,3})', r'\\\1_{\2}', result)

        result = re.sub(r'\b([a-zA-Z]{2,})_([a-zA-Z0-9]{1,3})\b', r'\1_{\2}', result)

        result = re.sub(r'\b([a-zA-Z]+)_(\d{1,3})\b', r'\1_{\2}', result)

        result = re.sub(r'\(([a-zA-Z]+)\)', r'(\1)', result)

        result = re.sub(r'\(([a-zA-Z]+)\s*\\cdot\s*([a-zA-Z]+)\)', r'(\1 \\cdot \2)', result)

        result = re.sub(r'(\w+)_(\d{1,3})(?=\s|[,;\)\.]|$)', r'\1_{\2}', result)

        result = re.sub(r'\b(\w+)\s*/\s*(\w+)\b', r'\\frac{\1}{\2}', result)

        result = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\\frac{\1}{\2}', result)

        return result

    def _clean_text(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = []
        prev_line = ""
        skip_next_empty = False
        in_markdown_block = False

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            if line_stripped.startswith('```'):
                if in_markdown_block:
                    in_markdown_block = False
                    continue
                else:
                    in_markdown_block = True
                    continue

            if in_markdown_block:
                cleaned_lines.append(line_stripped)
                continue

            if not line_stripped:
                skip_next_empty = False
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue

            if skip_next_empty and not line_stripped:
                continue

            if line_stripped == prev_line and len(line_stripped) > 30:
                skip_next_empty = True
                continue

            if i < len(lines) - 1:
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if line_stripped == next_line and len(line_stripped) < 50:
                    skip_next_empty = True
                    continue

            if self._is_header_footer(line_stripped):
                skip_next_empty = True
                continue

            if self._is_author_info(line_stripped):
                skip_next_empty = True
                continue

            cleaned_lines.append(line_stripped)
            prev_line = line_stripped
            skip_next_empty = False

        while cleaned_lines and cleaned_lines[-1] == "":
            cleaned_lines.pop()

        cleaned_text = "\n".join(cleaned_lines)

        cleaned_text = self._convert_to_latex(cleaned_text)

        return cleaned_text

    def process(self) -> Dict[str, Any]:
        documents = self.load_documents()

        all_text = ""

        for doc_data in documents:
            text = doc_data.get("text", "")
            cleaned_text = self._clean_text(text)
            all_text += cleaned_text + "\n\n"

        file_name = self.documents_dir.stem.replace("_documents", "")
        md_file = self.output_dir / f"{file_name}.md"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(all_text)

        logger.info(f"清洗完成:")
        logger.info(f"  - MD文件: {md_file}")
        logger.info(f"  - 文本长度: {len(all_text)}")

        return {
            "md_file": str(md_file),
            "text_length": len(all_text),
            "documents_count": len(documents)
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python clean_to_md.py <documents_dir>")
        print("示例: python clean_to_md.py ./parsed_pdf/2025EAI_documents")
        sys.exit(1)

    documents_dir = sys.argv[1]

    cleaner = DocumentCleaner(documents_dir)
    result = cleaner.process()

    print(f"\n处理完成!")
    print(f"MD文件: {result['md_file']}")
    print(f"文本长度: {result['text_length']}")
    print(f"页数: {result['documents_count']}")
