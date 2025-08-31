#!/usr/bin/env python3
"""
Template Manager - 模板管理器
功能：管理各种文章模板
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class TemplateManager:
    """模板管理器"""
    
    def __init__(self, template_dir: str = "Templates", config_path: str = "Config/template_config.yaml"):
        # 处理路径问题 - 使用绝对路径
        if not Path(template_dir).is_absolute():
            template_dir = Path(__file__).parent.parent / template_dir
        if not Path(config_path).is_absolute():
            config_path = Path(__file__).parent.parent / config_path
            
        self.template_dir = Path(template_dir)
        self.config = self._load_config(config_path)
        self.templates = self._discover_templates()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        return {
            "template_versions": {
                "latex": "1.0",
                "markdown": "1.0"
            },
            "backup_enabled": True,
            "backup_dir": "Templates/backups",
            "supported_types": ["latex", "markdown"],
            "template_variables": {
                "title": "文章标题",
                "author": "作者",
                "date": "日期",
                "subject": "主题",
                "abstract": "摘要",
                "keywords": "关键词",
                "introduction": "引言",
                "content": "主要内容",
                "conclusion": "结论",
                "references": "参考文献"
            }
        }
    
    def _discover_templates(self) -> Dict[str, Dict[str, Any]]:
        """发现所有模板"""
        templates = {}
        
        if not self.template_dir.exists():
            return templates
        
        for template_file in self.template_dir.glob("*.tex"):
            templates[template_file.stem] = {
                "path": template_file,
                "type": "latex",
                "version": self.config["template_versions"].get("latex", "1.0"),
                "created": datetime.fromtimestamp(template_file.stat().st_ctime),
                "modified": datetime.fromtimestamp(template_file.stat().st_mtime)
            }
        
        for template_file in self.template_dir.glob("*.md"):
            templates[template_file.stem] = {
                "path": template_file,
                "type": "markdown",
                "version": self.config["template_versions"].get("markdown", "1.0"),
                "created": datetime.fromtimestamp(template_file.stat().st_ctime),
                "modified": datetime.fromtimestamp(template_file.stat().st_mtime)
            }
        
        return templates
    
    def get_template(self, template_name: str) -> Optional[Path]:
        """获取模板文件路径"""
        if template_name in self.templates:
            return self.templates[template_name]["path"]
        
        # 尝试按类型获取
        for name, template_info in self.templates.items():
            if template_name.lower() in name.lower():
                return template_info["path"]
        
        return None
    
    def get_template_by_type(self, template_type: str) -> Optional[Path]:
        """根据类型获取默认模板"""
        template_map = {
            "latex": "latex-template",
            "markdown": "markdown-template"
        }
        
        default_name = template_map.get(template_type)
        if default_name:
            return self.get_template(default_name)
        
        # 如果没有默认模板，返回该类型的第一个模板
        for name, template_info in self.templates.items():
            if template_info["type"] == template_type:
                return template_info["path"]
        
        return None
    
    def render_template(self, template_path: Path, params: Dict[str, str]) -> str:
        """渲染模板"""
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # 简单的模板替换
        for key, value in params.items():
            template_content = template_content.replace(f"{{{{{key}}}}}", str(value))
        
        return template_content
    
    def create_template(self, name: str, template_type: str, content: str, 
                       description: str = "") -> Dict[str, Any]:
        """创建新模板"""
        try:
            if template_type not in self.config["supported_types"]:
                return {"success": False, "error": f"不支持的模板类型: {template_type}"}
            
            # 确定文件扩展名
            ext = ".tex" if template_type == "latex" else ".md"
            template_path = self.template_dir / f"{name}{ext}"
            
            # 检查是否已存在
            if template_path.exists():
                return {"success": False, "error": f"模板已存在: {name}"}
            
            # 创建备份
            if self.config["backup_enabled"]:
                self._backup_existing_templates()
            
            # 保存模板
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 更新模板列表
            self.templates[name] = {
                "path": template_path,
                "type": template_type,
                "version": "1.0",
                "description": description,
                "created": datetime.now(),
                "modified": datetime.now()
            }
            
            return {
                "success": True,
                "template_path": str(template_path),
                "message": f"成功创建模板: {name}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def update_template(self, name: str, content: str) -> Dict[str, Any]:
        """更新模板"""
        try:
            if name not in self.templates:
                return {"success": False, "error": f"模板不存在: {name}"}
            
            template_path = self.templates[name]["path"]
            
            # 创建备份
            if self.config["backup_enabled"]:
                self._backup_template(template_path)
            
            # 更新模板
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 更新信息
            self.templates[name]["modified"] = datetime.now()
            
            return {
                "success": True,
                "template_path": str(template_path),
                "message": f"成功更新模板: {name}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_template(self, name: str) -> Dict[str, Any]:
        """删除模板"""
        try:
            if name not in self.templates:
                return {"success": False, "error": f"模板不存在: {name}"}
            
            template_path = self.templates[name]["path"]
            
            # 创建备份
            if self.config["backup_enabled"]:
                self._backup_template(template_path)
            
            # 删除模板
            template_path.unlink()
            
            # 从列表中移除
            del self.templates[name]
            
            return {
                "success": True,
                "message": f"成功删除模板: {name}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_templates(self, template_type: str = None) -> Dict[str, Any]:
        """列出所有模板"""
        templates = self.templates
        
        if template_type:
            templates = {k: v for k, v in templates.items() if v["type"] == template_type}
        
        return {
            "success": True,
            "templates": templates,
            "count": len(templates)
        }
    
    def get_template_info(self, name: str) -> Dict[str, Any]:
        """获取模板信息"""
        if name not in self.templates:
            return {"success": False, "error": f"模板不存在: {name}"}
        
        template_info = self.templates[name].copy()
        template_info["variables"] = self._extract_template_variables(name)
        
        return {
            "success": True,
            "template_info": template_info
        }
    
    def _extract_template_variables(self, name: str) -> List[str]:
        """提取模板中的变量"""
        if name not in self.templates:
            return []
        
        template_path = self.templates[name]["path"]
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取 {{variable}} 格式的变量
        import re
        variables = re.findall(r'\{\{([^}]+)\}\}', content)
        
        return list(set(variables))
    
    def _backup_template(self, template_path: Path) -> None:
        """备份单个模板"""
        backup_dir = Path(self.config["backup_dir"])
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{template_path.stem}_{timestamp}{template_path.suffix}"
        
        shutil.copy2(template_path, backup_path)
    
    def _backup_existing_templates(self) -> None:
        """备份所有现有模板"""
        backup_dir = Path(self.config["backup_dir"])
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"templates_backup_{timestamp}"
        
        if self.template_dir.exists():
            shutil.copytree(self.template_dir, backup_path)
    
    def validate_template(self, name: str) -> Dict[str, Any]:
        """验证模板"""
        if name not in self.templates:
            return {"success": False, "error": f"模板不存在: {name}"}
        
        template_path = self.templates[name]["path"]
        template_type = self.templates[name]["type"]
        
        issues = []
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查基本结构
            if template_type == "latex":
                if "\\documentclass" not in content:
                    issues.append("缺少\\documentclass声明")
                if "\\begin{document}" not in content:
                    issues.append("缺少\\begin{document}标签")
                if "\\end{document}" not in content:
                    issues.append("缺少\\end{document}标签")
            
            # 检查变量使用
            variables = self._extract_template_variables(name)
            required_vars = self.config["template_variables"]
            
            for var in required_vars:
                if var not in variables:
                    issues.append(f"缺少推荐变量: {var}")
            
            return {
                "success": True,
                "is_valid": len(issues) == 0,
                "issues": issues,
                "variables": variables
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# 使用示例
if __name__ == "__main__":
    manager = TemplateManager()
    
    # 列出所有模板
    print("所有模板:")
    print(json.dumps(manager.list_templates(), ensure_ascii=False, indent=2))
    
    # 获取模板信息
    print("\nLaTeX模板信息:")
    print(json.dumps(manager.get_template_info("latex-template"), ensure_ascii=False, indent=2))