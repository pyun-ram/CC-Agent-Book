#!/usr/bin/env python3
"""
BookChapter Agent - 文章创建智能体
功能：根据用户指定的主题创建标准化文章
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import re

try:
    from template_manager import TemplateManager
except ImportError:
    # 当从其他目录导入时，添加当前目录到路径
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from template_manager import TemplateManager


class BookChapterAgent:
    def __init__(self, config_path: str = "Config/agent_config.yaml"):
        # 处理配置文件路径
        if not Path(config_path).is_absolute():
            config_path = Path(__file__).parent.parent / config_path
        self.config = self._load_config(config_path)
        self.template_manager = TemplateManager()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        # 简化配置，直接使用默认值
        # 注意：实际应该从配置文件加载，这里为了简化使用默认值
        return {
            "book_folder": str(Path(__file__).parent.parent / "Book"),
            "default_author": "AI System",
            "date_format": "%Y%m%d",
            "supported_formats": ["latex", "markdown"]
        }
    
    def create_article(self, topic: str, file_type: str, filename: str, 
                     author: str = None, custom_params: Dict[str, str] = None) -> Dict[str, Any]:
        """
        创建新文章
        
        Args:
            topic: 文章主题
            file_type: 文件类型 (latex/markdown)
            filename: 文件名
            author: 作者 (可选)
            custom_params: 自定义参数 (可选)
            
        Returns:
            Dict: 创建结果
        """
        try:
            # 验证输入
            if file_type not in self.config["supported_formats"]:
                return {"success": False, "error": f"不支持的文件类型: {file_type}"}
            
            # 生成文件夹名
            folder_name = self._generate_folder_name(topic, filename)
            folder_path = Path(self.config["book_folder"]) / folder_name
            
            # 创建文件夹
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # 选择模板
            template_path = self.template_manager.get_template_by_type(file_type)
            if not template_path:
                return {"success": False, "error": f"找不到{file_type}模板"}
            
            # 生成文章内容
            content_params = self._prepare_content_params(topic, author, custom_params)
            article_content = self.template_manager.render_template(template_path, content_params)
            
            # 保存文件
            file_ext = ".tex" if file_type == "latex" else ".md"
            file_path = folder_path / f"{filename}{file_ext}"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(article_content)
            
            return {
                "success": True,
                "folder_path": str(folder_path),
                "file_path": str(file_path),
                "folder_name": folder_name,
                "message": f"成功创建{file_type}文章: {filename}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_folder_name(self, topic: str, filename: str) -> str:
        """生成文件夹名"""
        date_str = datetime.now().strftime(self.config["date_format"])
        # 简化主题用于文件夹名
        simple_topic = re.sub(r'[^\w\s-]', '', topic).replace(' ', '-')[:20]
        return f"{date_str}-{simple_topic}"
    
    def _prepare_content_params(self, topic: str, author: str = None, 
                               custom_params: Dict[str, str] = None) -> Dict[str, str]:
        """准备内容参数"""
        params = {
            "title": topic,
            "author": author or self.config["default_author"],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "subject": topic,
            "abstract": f"本文主要讨论{topic}的相关内容",
            "keywords": topic.replace(' ', ','),
            "introduction": f"本文旨在探讨{topic}的基本概念和应用。",
            "content": f"# {topic}\n\n## 基本概念\n\n{topic}是一个重要的研究领域。\n\n## 主要内容\n\n### 定义\n\n### 应用\n\n### 发展趋势\n\n",
            "conclusion": f"通过本文的讨论，我们对{topic}有了更深入的理解。",
            "references": "[1] 相关文献"
        }
        
        # 合并自定义参数
        if custom_params:
            params.update(custom_params)
            
        return params




# 使用示例
if __name__ == "__main__":
    agent = BookChapterAgent()
    result = agent.create_article(
        topic="SVD分解",
        file_type="latex", 
        filename="SVD-Decomposition",
        author="张三"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))