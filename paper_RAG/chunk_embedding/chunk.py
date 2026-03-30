import re
import json
from pathlib import Path
from typing import List, Dict, Any


class DocumentChunker:
    def __init__(self, json_file: str, chunk_size: int = 500, overlap: int = 50):
        self.json_file = Path(json_file)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.data: Dict[str, Any] = {}
        self.chunks: List[Dict[str, Any]] = []

    def load_json(self) -> Dict[str, Any]:
        with open(self.json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def _split_into_sentences(self, text: str) -> List[str]:
        abbreviations = ['et al', 'e.g', 'i.e', 'fig', 'vs', 'etc', 'sr', 'dr', 'mr', 'mrs', 'ms', 'inc', 'ltd', 'corp']
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+(?=[A-Z]|$)'

        def replace_with_placeholder(match):
            placeholder = f"__SPLIT_{len([m for m in re.finditer(r'__SPLIT_', text[:match.start()])])}_"
            return placeholder

        text_protected = text
        for abbr in abbreviations:
            text_protected = re.sub(rf'\b{abbr}\.\s+(?=[a-z])', abbr + '.|||SPLIT_MARKER|||', text_protected, flags=re.IGNORECASE)

        text_protected = re.sub(r'([.!?])\s+', r'\1|||', text_protected)
        text_protected = text_protected.replace('|||SPLIT_MARKER|||', ' ')

        sentences = text_protected.split('|||')
        result = []
        for s in sentences:
            s = s.strip()
            if s:
                s = re.sub(r'__SPLIT_\d+_', '. ', s)
                result.append(s)

        return result

    def _chunk_into_parents(self, text: str, heading_info: Dict[str, str]) -> List[Dict[str, Any]]:
        if not text or len(text) == 0:
            return []

        code_blocks, text = self._extract_code_blocks(text)
        tables, text = self._extract_tables(text)

        sentences = self._split_into_sentences(text)
        parent_chunks = []
        current_sentences = []
        current_size = 0

        section_prefix = ""
        if heading_info:
            parts = [heading_info.get(k, "") for k in ["h1_title", "h2_title", "h3_title"] if heading_info.get(k)]
            if parts:
                section_prefix = " | ".join(parts) + "\n\n"

        parent_size = 2000

        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size + 1 <= parent_size:
                current_sentences.append(sentence)
                current_size = len(section_prefix) + len(" ".join(current_sentences))
            else:
                if current_sentences:
                    chunk_text = section_prefix + " ".join(current_sentences)
                    for cb in code_blocks[:]:
                        if len(cb.get("content", "")) <= parent_size * 0.3:
                            chunk_text += f"\n\n```{cb.get('language', '')}\n{cb['content']}\n```"
                            code_blocks.remove(cb)
                    for tb in tables[:]:
                        if len(tb.get("content", "")) <= parent_size * 0.3:
                            chunk_text += f"\n\n[表格]\n{tb['content']}"
                            tables.remove(tb)
                    parent_chunks.append({
                        "content": chunk_text,
                        "chunk_size": len(chunk_text)
                    })

                overlap_sentences = []
                overlap_size = 0
                if self.overlap > 0:
                    for s in reversed(current_sentences):
                        if overlap_size + len(s) + 1 <= self.overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s) + 1
                        else:
                            break

                if overlap_sentences:
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_sentences = [sentence]
                current_size = len(section_prefix) + len(" ".join(current_sentences))

        if current_sentences:
            chunk_text = section_prefix + " ".join(current_sentences)
            for cb in code_blocks:
                chunk_text += f"\n\n```{cb.get('language', '')}\n{cb['content']}\n```"
            for tb in tables:
                chunk_text += f"\n\n[表格]\n{tb['content']}"
            parent_chunks.append({
                "content": chunk_text,
                "chunk_size": len(chunk_text)
            })

        return parent_chunks

    def _chunk_parent_into_children(self, parent_text: str, parent_index: int, heading_info: Dict[str, str]) -> List[Dict[str, Any]]:
        child_size = 500
        sentences = self._split_into_sentences(parent_text)
        children = []
        current_sentences = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size + 1 <= child_size:
                current_sentences.append(sentence)
                current_size = len(" ".join(current_sentences))
            else:
                if current_sentences:
                    children.append({
                        "content": " ".join(current_sentences),
                        "chunk_size": len(" ".join(current_sentences)),
                        "parent_index": parent_index,
                        "parent_content": parent_text[:500] if len(parent_text) > 500 else parent_text
                    })

                overlap_sentences = []
                overlap_size = 0
                if self.overlap > 0:
                    for s in reversed(current_sentences):
                        if overlap_size + len(s) + 1 <= self.overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s) + 1
                        else:
                            break

                if overlap_sentences:
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_sentences = [sentence]
                current_size = len(" ".join(current_sentences))

        if current_sentences:
            children.append({
                "content": " ".join(current_sentences),
                "chunk_size": len(" ".join(current_sentences)),
                "parent_index": parent_index,
                "parent_content": parent_text[:500] if len(parent_text) > 500 else parent_text
            })

        return children

    def _restore_placeholders(self, text: str, code_blocks: List[Dict], tables: List[Dict]) -> str:
        for i, cb in enumerate(code_blocks):
            lang = cb.get("language", "")
            text = text.replace(f"__CODE_BLOCK_{i}__", f"```{lang}\n{cb['content']}\n```")
        for i, table in enumerate(tables):
            text = text.replace(f"__TABLE_{i}__", f"[表格]\n{table['content']}")
        return text

    def _extract_code_blocks(self, text: str) -> tuple:
        code_blocks = []
        text_parts = []
        in_fence = False
        fence_lang = ""
        current_fence = []

        for line in text.split('\n'):
            if line.strip().startswith('```') and not in_fence:
                in_fence = True
                fence_lang = line.strip()[3:].strip()
                current_fence = []
            elif line.strip().startswith('```') and in_fence:
                code_blocks.append({
                    "content": '\n'.join(current_fence),
                    "language": fence_lang
                })
                text_parts.append(f"__CODE_BLOCK_{len(code_blocks)-1}__")
                current_fence = []
                in_fence = False
                fence_lang = ""
            elif in_fence:
                current_fence.append(line)
            else:
                text_parts.append(line)

        return code_blocks, '\n'.join(text_parts)

    def _extract_tables(self, text: str) -> tuple:
        tables = []
        text_parts = []
        lines = text.split('\n')
        i = 0

        while i < len(lines):
            if '|' in lines[i] and re.search(r'\|[-:\s]+\|', lines[i]):
                table_lines = [lines[i]]
                j = i + 1
                while j < len(lines) and '|' in lines[j]:
                    table_lines.append(lines[j])
                    j += 1
                tables.append({
                    "content": '\n'.join(table_lines),
                    "header": table_lines[0] if table_lines else ""
                })
                text_parts.append(f"__TABLE_{len(tables)-1}__")
                i = j
            else:
                text_parts.append(lines[i])
                i += 1

        return tables, '\n'.join(text_parts)

    def _get_parent_heading(self, node: Dict[str, Any], heading_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        for h in heading_list:
            if h["text"] == node["text"] and h["line_number"] == node["line_number"]:
                idx = heading_list.index(h)
                for i in range(idx - 1, -1, -1):
                    if heading_list[i]["level"] == 1:
                        return heading_list[i]
        return {"text": "", "h_level": "H1"}

    def process(self) -> List[Dict[str, Any]]:
        self.load_json()
        tree = self.data.get("tree", {})
        children = tree.get("children", [])

        for h1_node in children:
            h1_text = h1_node.get("text", "")
            h1_h_level = h1_node.get("h_level", "H1")
            h1_content = h1_node.get("content", "")

            heading_info = {"h1_title": h1_text, "h2_title": "", "h3_title": ""}

            if h1_content and len(h1_content) > 2000:
                parent_chunks = self._chunk_into_parents(h1_content, heading_info)
                for idx, parent in enumerate(parent_chunks):
                    parent_idx = len(self.chunks)
                    children_chunks = self._chunk_parent_into_children(parent["content"], parent_idx, heading_info)
                    self.chunks.append({
                        "h1_title": h1_text,
                        "h2_title": "",
                        "h3_title": "",
                        "h1_level": h1_h_level,
                        "h2_level": "",
                        "h3_level": "",
                        "content": parent["content"],
                        "sub_chunk": parent["content"][:500] if len(parent["content"]) > 500 else parent["content"],
                        "parent_content": parent["content"],
                        "chunk_size": parent["chunk_size"],
                        "is_parent": True
                    })
                    for child in children_chunks:
                        self.chunks.append({
                            "h1_title": h1_text,
                            "h2_title": "",
                            "h3_title": "",
                            "h1_level": h1_h_level,
                            "h2_level": "",
                            "h3_level": "",
                            "content": child["content"],
                            "sub_chunk": child["content"],
                            "parent_content": parent["content"],
                            "parent_index": parent_idx,
                            "chunk_size": child["chunk_size"],
                            "is_parent": False
                        })

            h1_children = h1_node.get("children", [])
            for h2_node in h1_children:
                h2_text = h2_node.get("text", "")
                h2_h_level = h2_node.get("h_level", "H2")
                h2_content = h2_node.get("content", "")

                heading_info = {"h1_title": h1_text, "h2_title": h2_text, "h3_title": ""}

                if h2_content and len(h2_content) > 2000:
                    parent_chunks = self._chunk_into_parents(h2_content, heading_info)
                    for idx, parent in enumerate(parent_chunks):
                        parent_idx = len(self.chunks)
                        children_chunks = self._chunk_parent_into_children(parent["content"], parent_idx, heading_info)
                        self.chunks.append({
                            "h1_title": h1_text,
                            "h2_title": h2_text,
                            "h3_title": "",
                            "h1_level": h1_h_level,
                            "h2_level": h2_h_level,
                            "h3_level": "",
                            "content": parent["content"],
                            "sub_chunk": parent["content"][:500] if len(parent["content"]) > 500 else parent["content"],
                            "parent_content": parent["content"],
                            "chunk_size": parent["chunk_size"],
                            "is_parent": True
                        })
                        for child in children_chunks:
                            self.chunks.append({
                                "h1_title": h1_text,
                                "h2_title": h2_text,
                                "h3_title": "",
                                "h1_level": h1_h_level,
                                "h2_level": h2_h_level,
                                "h3_level": "",
                                "content": child["content"],
                                "sub_chunk": child["content"],
                                "parent_content": parent["content"],
                                "parent_index": parent_idx,
                                "chunk_size": child["chunk_size"],
                                "is_parent": False
                            })

                h2_children = h2_node.get("children", [])
                for h3_node in h2_children:
                    h3_text = h3_node.get("text", "")
                    h3_h_level = h3_node.get("h_level", "H3")
                    h3_content = h3_node.get("content", "")

                    heading_info = {"h1_title": h1_text, "h2_title": h2_text, "h3_title": h3_text}

                    if h3_content and len(h3_content) > 2000:
                        parent_chunks = self._chunk_into_parents(h3_content, heading_info)
                        for idx, parent in enumerate(parent_chunks):
                            parent_idx = len(self.chunks)
                            children_chunks = self._chunk_parent_into_children(parent["content"], parent_idx, heading_info)
                            self.chunks.append({
                                "h1_title": h1_text,
                                "h2_title": h2_text,
                                "h3_title": h3_text,
                                "h1_level": h1_h_level,
                                "h2_level": h2_h_level,
                                "h3_level": h3_h_level,
                                "content": parent["content"],
                                "sub_chunk": parent["content"][:500] if len(parent["content"]) > 500 else parent["content"],
                                "parent_content": parent["content"],
                                "chunk_size": parent["chunk_size"],
                                "is_parent": True
                            })
                            for child in children_chunks:
                                self.chunks.append({
                                    "h1_title": h1_text,
                                    "h2_title": h2_text,
                                    "h3_title": h3_text,
                                    "h1_level": h1_h_level,
                                    "h2_level": h2_h_level,
                                    "h3_level": h3_h_level,
                                    "content": child["content"],
                                    "sub_chunk": child["content"],
                                    "parent_content": parent["content"],
                                    "parent_index": parent_idx,
                                    "chunk_size": child["chunk_size"],
                                    "is_parent": False
                                })

                    elif h3_content:
                        self.chunks.append({
                            "h1_title": h1_text,
                            "h2_title": h2_text,
                            "h3_title": h3_text,
                            "h1_level": h1_h_level,
                            "h2_level": h2_h_level,
                            "h3_level": h3_h_level,
                            "content": h3_content,
                            "sub_chunk": h3_content[:500] if len(h3_content) > 500 else h3_content,
                            "parent_content": h3_content[:500] if len(h3_content) > 500 else h3_content,
                            "chunk_size": len(h3_content),
                            "is_parent": True
                        })
                    elif h3_text:
                        pass

                if not h2_children and h2_content:
                    self.chunks.append({
                        "h1_title": h1_text,
                        "h2_title": h2_text,
                        "h3_title": "",
                        "h1_level": h1_h_level,
                        "h2_level": h2_h_level,
                        "h3_level": "",
                        "content": h2_content,
                        "sub_chunk": h2_content[:500] if len(h2_content) > 500 else h2_content,
                        "parent_content": h2_content[:500] if len(h2_content) > 500 else h2_content,
                        "chunk_size": len(h2_content),
                        "is_parent": True
                    })

            if not h1_children and h1_content:
                self.chunks.append({
                    "h1_title": h1_text,
                    "h2_title": "",
                    "h3_title": "",
                    "h1_level": h1_h_level,
                    "h2_level": "",
                    "h3_level": "",
                    "content": h1_content,
                    "sub_chunk": h1_content[:500] if len(h1_content) > 500 else h1_content,
                    "parent_content": h1_content[:500] if len(h1_content) > 500 else h1_content,
                    "chunk_size": len(h1_content),
                    "is_parent": True
                })

        return self.chunks

    def save_chunks(self, output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_name = self.json_file.stem.replace("_headings", "")
        output_file = output_path / f"{file_name}_chunks.json"

        result = {
            "file_name": file_name,
            "total_chunks": len(self.chunks),
            "chunks": self.chunks
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return str(output_file)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python chunk.py <headings_json> [chunk_size] [overlap]")
        print("示例: python chunk.py ./parsed_pdf/2025EAIv10_headings.json 500 50")
        sys.exit(1)

    json_file = sys.argv[1]
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    chunker = DocumentChunker(json_file, chunk_size, overlap)
    chunks = chunker.process()

    output_dir = r"D:\AI\paper_RAG\chunk_embedding\chunked_json"
    output_file = chunker.save_chunks(output_dir)

    print(f"\n处理完成!")
    print(f"总块数: {len(chunks)}")
    print(f"输出文件: {output_file}")
