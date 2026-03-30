import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class MDHeadingParser:
    def __init__(self, md_file: str):
        self.md_file = Path(md_file)
        self.headings: List[Dict[str, Any]] = []
        self.tree: Dict[str, Any] = {}

    def load_md(self) -> str:
        with open(self.md_file, "r", encoding="utf-8") as f:
            return f.read()

    def _get_heading_level(self, line: str) -> int:
        line_stripped = line.strip()

        match = re.match(r'^(#{1,3})\s+(.+)$', line_stripped)
        if not match:
            return 0

        hash_count = len(match.group(1))
        content = match.group(2).strip()

        if hash_count == 1:
            if re.match(r'^\d+\.\d+\s+', content):
                return 2
            return 1
        elif hash_count == 2:
            if re.match(r'^\d+\.\d+\s+', content):
                return 3
            return 2
        elif hash_count == 3:
            return 3

        return 0

    def _get_heading_text(self, line: str) -> str:
        match = re.match(r'^#+\s+(.+)$', line.strip())
        if match:
            return match.group(1).strip()
        return ""

    def find_headings(self, content: str) -> List[Dict[str, Any]]:
        lines = content.split('\n')
        headings = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith('#'):
                level = self._get_heading_level(line_stripped)
                if level > 0:
                    text = self._get_heading_text(line_stripped)
                    headings.append({
                        "level": level,
                        "text": text,
                        "line_number": i + 1,
                        "h_level": f"H{level}"
                    })

        self.headings = headings
        return headings

    def build_tree_with_content(self) -> Dict[str, Any]:
        if not self.headings:
            return {}

        lines = self.load_md().split('\n')

        root: Dict[str, Any] = {"children": []}
        stack = [root]

        for idx, heading in enumerate(self.headings):
            start_line = heading["line_number"] - 1
            end_line = self.headings[idx + 1]["line_number"] - 1 if idx + 1 < len(self.headings) else len(lines)

            content_lines = []
            for i in range(start_line + 1, end_line):
                line = lines[i].strip()
                if line:
                    content_lines.append(line)
            content = "\n".join(content_lines)

            node = {
                "level": heading["level"],
                "text": heading["text"],
                "line_number": heading["line_number"],
                "h_level": heading.get("h_level", f"H{heading['level']}"),
                "content": content,
                "children": []
            }

            while len(stack) > 1 and stack[-1]["level"] >= heading["level"]:
                stack.pop()

            stack[-1]["children"].append(node)
            stack.append(node)

        self.tree = root
        return root

    def _tree_to_list(self, node: Dict[str, Any], level: int = 1) -> List[Dict[str, Any]]:
        result = []
        for child in node.get("children", []):
            result.append({
                "level": child["level"],
                "text": child["text"],
                "line_number": child["line_number"],
                "h_level": f"H{child['level']}"
            })
            result.extend(self._tree_to_list(child, level + 1))
        return result

    def get_flat_heading_list(self) -> List[Dict[str, Any]]:
        flat_list = []
        for heading in self.headings:
            flat_list.append({
                "level": heading["level"],
                "text": heading["text"],
                "line_number": heading["line_number"],
                "h_level": f"H{heading['level']}"
            })
        return flat_list

    def process(self) -> Dict[str, Any]:
        content = self.load_md()
        self.find_headings(content)
        self.build_tree_with_content()

        file_name = self.md_file.stem

        return {
            "file_name": file_name,
            "total_headings": len(self.headings),
            "headings": self.get_flat_heading_list(),
            "tree": self.tree
        }

    def save_json(self, output_path: Optional[str] = None) -> str:
        result = self.process()

        if output_path is None:
            output_path = self.md_file.parent / f"{self.md_file.stem}_headings.json"
        else:
            output_path = Path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return str(output_path)

    def _print_tree_node(self, node: Dict[str, Any], indent: int = 0):
        for child in node.get("children", []):
            prefix = "  " * indent
            print(f"{prefix}{child['h_level']}: {child['text']}")
            self._print_tree_node(child, indent + 1)

    def print_tree(self):
        print("\n=== 标题列表 ===")
        for h in self.headings:
            indent = "  " * (h["level"] - 1)
            print(f"{indent}{h['h_level']}: {h['text']}")

        print("\n=== 层级树 ===")
        self._print_tree_node(self.tree)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python md_to_json.py <md_file>")
        print("示例: python md_to_json.py ./parsed_pdf/2025EAIv10.md")
        sys.exit(1)

    md_file = sys.argv[1]

    parser = MDHeadingParser(md_file)
    result = parser.process()

    parser.print_tree()

    json_path = parser.save_json()
    print(f"\n处理完成!")
    print(f"标题总数: {result['total_headings']}")
    print(f"JSON文件: {json_path}")
