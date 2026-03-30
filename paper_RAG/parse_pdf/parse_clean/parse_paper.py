import os
import json

from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger


class DocumentParser:
    def __init__(
        self,
        output_dir: str = "./output",
        llamaparse_api_key: str = None,
        use_proxy: bool = False,
        proxy_url: str = None
    ):
        self.output_dir = Path(output_dir)
        self.llamaparse_api_key = llamaparse_api_key or os.getenv("LLAMAPARSE_API_KEY")
        if not self.llamaparse_api_key:
            raise ValueError("需要设置 LLAMAPARSE_API_KEY")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_proxy = use_proxy
        self.proxy_url = proxy_url or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")

    def parse_pdf(self, pdf_path: str) -> dict:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        file_name = pdf_path.stem
        logger.info(f"开始解析 PDF: {pdf_path}")

        documents = self._parse_with_llamaparse(pdf_path)
        logger.info(f"LlamaParse 解析完成, 共 {len(documents)} 页")

        documents_dir = self.output_dir / f"{file_name}_documents"
        documents_dir.mkdir(parents=True, exist_ok=True)

        for i, doc in enumerate(documents):
            doc_file = documents_dir / f"page_{i+1}.json"
            doc_data = {
                "page": i + 1,
                "text": doc.text,
                "metadata": doc.metadata
            }
            with open(doc_file, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=2)

        logger.info(f"PDF 解析完成: {file_name}")
        logger.info(f"输出目录: {documents_dir}")

        return {
            "file_name": file_name,
            "output_dir": str(self.output_dir),
            "documents_dir": str(documents_dir),
            "documents_count": len(documents)
        }

    def _parse_with_llamaparse(self, pdf_path: Path):
        saved_http = os.environ.get("HTTP_PROXY")
        saved_https = os.environ.get("HTTPS_PROXY")

        if self.use_proxy and self.proxy_url:
            os.environ["HTTP_PROXY"] = self.proxy_url
            os.environ["HTTPS_PROXY"] = self.proxy_url
            logger.info(f"设置代理: {self.proxy_url}")

        try:
            from llama_parse import LlamaParse
        except ImportError:
            if self.use_proxy and self.proxy_url:
                os.environ["HTTP_PROXY"] = saved_http or ""
                os.environ["HTTPS_PROXY"] = saved_https or ""
            raise ImportError("请安装 llama-parse: pip install llama-parse")

        logger.info(f"LlamaParse API key: {self.llamaparse_api_key[:10]}..." if self.llamaparse_api_key else "API key is None")

        parser = LlamaParse(
            api_key=self.llamaparse_api_key,
            result_type="markdown",
            num_workers=4,
            verbose=True,
            skip_diagrams=True,
        )

        try:
            documents = parser.load_data(str(pdf_path))
            logger.info(f"LlamaParse loaded {len(documents)} documents")
        except Exception as e:
            logger.error(f"LlamaParse load_data failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            if self.use_proxy and self.proxy_url:
                if saved_http:
                    os.environ["HTTP_PROXY"] = saved_http
                else:
                    os.environ.pop("HTTP_PROXY", None)
                if saved_https:
                    os.environ["HTTPS_PROXY"] = saved_https
                else:
                    os.environ.pop("HTTPS_PROXY", None)

        return documents


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python parse_paper.py <pdf_path> [output_dir]")
        print("示例: python parse_paper.py D:\\桌面\\论文整理\\2025EAI.pdf ./parsed_pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./parsed_pdf"

    parser = DocumentParser(output_dir=output_dir)
    result = parser.parse_pdf(pdf_path)

    print(f"\n解析完成!")
    print(f"PDF文件: {result['file_name']}")
    print(f"页数: {result['documents_count']}")
    print(f"输出目录: {result['documents_dir']}")
