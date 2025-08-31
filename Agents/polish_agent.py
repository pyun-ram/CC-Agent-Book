#!/usr/bin/env python3
"""
Polish Agent - 文章润色智能体
功能：对生成的文章进行格式化和润色
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


class PolishAgent:
    def __init__(self, config_path: str = "Config/agent_config.yaml"):
        # 处理配置文件路径
        if not Path(config_path).is_absolute():
            config_path = Path(__file__).parent.parent / config_path
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        return {
            "polish_options": {
                "format_normalization": True,
                "consistency_check": True,
                "grammar_check": True,
                "academic_polish": True
            },
            "supported_formats": ["latex", "markdown"],
            "max_file_size": 1024 * 1024  # 1MB
        }
    
    def polish_article(self, file_path: str, options: List[str] = None) -> Dict[str, Any]:
        """
        润色文章
        
        Args:
            file_path: 文章文件路径
            options: 润色选项列表
            
        Returns:
            Dict: 润色结果
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"success": False, "error": f"文件不存在: {file_path}"}
            
            # 检查文件大小
            if file_path.stat().st_size > self.config["max_file_size"]:
                return {"success": False, "error": "文件过大，超过1MB限制"}
            
            # 确定文件类型
            file_type = self._detect_file_type(file_path)
            if file_type not in self.config["supported_formats"]:
                return {"success": False, "error": f"不支持的文件类型: {file_type}"}
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 应用润色选项
            polish_options = options or list(self.config["polish_options"].keys())
            result = {"success": True, "changes": []}
            
            original_content = content
            
            for option in polish_options:
                if option in self.config["polish_options"]:
                    content, changes = self._apply_polish_option(content, option, file_type)
                    if changes:
                        result["changes"].extend(changes)
            
            # 保存润色后的内容
            backup_path = file_path.with_suffix(f".backup{file_path.suffix}")
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            result["backup_path"] = str(backup_path)
            result["message"] = f"成功润色文章，应用了{len(result['changes'])}项修改"
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _detect_file_type(self, file_path: Path) -> str:
        """检测文件类型"""
        suffix = file_path.suffix.lower()
        if suffix == ".tex":
            return "latex"
        elif suffix == ".md":
            return "markdown"
        else:
            return "unknown"
    
    def _apply_polish_option(self, content: str, option: str, file_type: str) -> tuple:
        """应用单个润色选项"""
        changes = []
        
        if option == "format_normalization":
            content, changes = self._normalize_format(content, file_type)
        elif option == "consistency_check":
            content, changes = self._check_consistency(content, file_type)
        elif option == "grammar_check":
            content, changes = self._check_grammar(content, file_type)
        elif option == "academic_polish":
            content, changes = self._academic_polish(content, file_type)
        
        return content, changes
    
    def _normalize_format(self, content: str, file_type: str) -> tuple:
        """格式规范化"""
        changes = []
        
        if file_type == "latex":
            # LaTeX格式规范化
            patterns = [
                (r'\s*\\begin\{([a-zA-Z]+)\}', r'\\begin{\1}', "环境标签格式"),
                (r'\s*\\end\{([a-zA-Z]+)\}', r'\\end{\1}', "环境结束标签格式"),
                (r'\\\\\s*\n', r'\\\\\n', "换行符格式"),
                (r'\$\s*([^$]+)\s*\$', r'$\1$', "数学公式空格")
            ]
        else:  # markdown
            # Markdown格式规范化
            patterns = [
                (r'^#+\s*', lambda m: m.group().upper(), "标题格式"),
                (r'\*\*([^*]+)\*\*', r'**\1**', "粗体格式"),
                (r'\*([^*]+)\*', r'*\1*', "斜体格式"),
                (r'\n{3,}', r'\n\n', "多余空行")
            ]
        
        for pattern, replacement, description in patterns:
            original_content = content
            if callable(replacement):
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            else:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            if content != original_content:
                changes.append(f"规范化: {description}")
        
        return content, changes
    
    def _check_consistency(self, content: str, file_type: str) -> tuple:
        """一致性检查"""
        changes = []
        
        # 检查术语一致性（简化版）
        common_terms = {
            "机器学习": ["Machine Learning", "Machine learning", "machine learning"],
            "深度学习": ["Deep Learning", "Deep learning", "deep learning"],
            "神经网络": ["Neural Network", "Neural network", "neural network"]
        }
        
        for chinese_term, variants in common_terms.items():
            # 检查是否有不同的英文表达
            found_variants = []
            for variant in variants:
                if variant in content:
                    found_variants.append(variant)
            
            if len(found_variants) > 1:
                # 统一为第一个变体
                unified_variant = found_variants[0]
                for variant in found_variants[1:]:
                    content = content.replace(variant, unified_variant)
                changes.append(f"术语统一: {chinese_term} -> {unified_variant}")
        
        return content, changes
    
    def _check_grammar(self, content: str, file_type: str) -> tuple:
        """语法检查（简化版）"""
        changes = []
        
        # 中文语法检查
        chinese_patterns = [
            (r'([。！？])\s*([a-zA-Z])', r'\1 \2', "中英文空格"),
            (r'([a-zA-Z])\s*([，。！？；：])', r'\1\2', "英文中文标点"),
            (r'\s{2,}', r' ', "多余空格"),
            (r'([^。！？])$', r'\1。', "句子结束标点")
        ]
        
        for pattern, replacement, description in chinese_patterns:
            original_content = content
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            if content != original_content:
                changes.append(f"语法修正: {description}")
        
        return content, changes
    
    def _academic_polish(self, content: str, file_type: str) -> tuple:
        """学术润色（简化版）"""
        changes = []
        
        # 学术表达优化
        academic_improvements = [
            (r'我们认为', r'研究表明', "学术表达"),
            (r'很明显', r'研究表明', "学术表达"),
            (r'很多', r'大量', "学术表达"),
            (r'一些', r'若干', "学术表达"),
            (r'好的', r'有效的', "学术表达"),
            (r'不好的', r'低效的', "学术表达")
        ]
        
        for informal, formal, description in academic_improvements:
            original_content = content
            content = content.replace(informal, formal)
            
            if content != original_content:
                changes.append(f"学术优化: {description}")
        
        return content, changes
    
    def get_polish_suggestions(self, content: str, file_type: str) -> Dict[str, Any]:
        """获取润色建议"""
        suggestions = {
            "format_issues": [],
            "consistency_issues": [],
            "grammar_issues": [],
            "academic_suggestions": []
        }
        
        # 分析内容并提供建议
        if file_type == "latex":
            if "\\begin{document}" not in content:
                suggestions["format_issues"].append("缺少\\begin{document}标签")
            if "\\end{document}" not in content:
                suggestions["format_issues"].append("缺少\\end{document}标签")
        
        # 检查基本问题
        if "  " in content:
            suggestions["format_issues"].append("存在多余空格")
        
        if len(content.split('\n')) < 10:
            suggestions["academic_suggestions"].append("文章较短，建议补充更多内容")
        
        return suggestions


# 使用示例
if __name__ == "__main__":
    agent = PolishAgent()
    
    # 示例：润色一个文件
    result = agent.polish_article(
        "Book/test-article.md",
        options=["format_normalization", "consistency_check", "grammar_check"]
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))