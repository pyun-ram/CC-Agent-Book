#!/usr/bin/env python3
"""
Workflow Orchestrator - 工作流协调器
功能：协调各个Agent的工作流程
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

try:
    from book_chapter_agent import BookChapterAgent
    from polish_agent import PolishAgent
    from template_manager import TemplateManager
except ImportError:
    # 当从其他目录导入时，添加当前目录到路径
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from book_chapter_agent import BookChapterAgent
    from polish_agent import PolishAgent
    from template_manager import TemplateManager


class WorkflowOrchestrator:
    """工作流协调器"""
    
    def __init__(self, config_path: str = "Config/agent_config.yaml"):
        # 处理配置文件路径
        if not Path(config_path).is_absolute():
            config_path = Path(__file__).parent.parent / config_path
        self.config = self._load_config(config_path)
        self.agents = self._initialize_agents()
        self.workflow_history = []
        self.current_workflow = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        return {
            "workflows": {
                "create_article": {
                    "steps": ["book_chapter", "polish"],
                    "quality_gates": True,
                    "retry_on_failure": True,
                    "max_retries": 3
                },
                "polish_only": {
                    "steps": ["polish"],
                    "quality_gates": True,
                    "retry_on_failure": True,
                    "max_retries": 2
                },
                "template_management": {
                    "steps": ["template_manager"],
                    "quality_gates": False,
                    "retry_on_failure": False
                }
            },
            "logging": {
                "enabled": True,
                "log_file": "Data/workflow_logs.json",
                "log_level": "INFO"
            },
            "quality_threshold": {
                "min_content_length": 100,
                "max_errors": 5
            }
        }
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """初始化所有Agent"""
        return {
            "book_chapter": BookChapterAgent(),
            "polish": PolishAgent(),
            "template_manager": TemplateManager()
        }
    
    def execute_workflow(self, workflow_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工作流程
        
        Args:
            workflow_name: 工作流名称
            params: 工作流参数
            
        Returns:
            Dict: 执行结果
        """
        try:
            if workflow_name not in self.config["workflows"]:
                return {"success": False, "error": f"未知的工作流: {workflow_name}"}
            
            workflow_config = self.config["workflows"][workflow_name]
            steps = workflow_config["steps"]
            
            # 记录工作流开始
            workflow_id = self._generate_workflow_id()
            self.current_workflow = {
                "id": workflow_id,
                "name": workflow_name,
                "params": params,
                "start_time": datetime.now(),
                "status": "running",
                "steps": [],
                "current_step": 0
            }
            
            self._log_workflow("START", f"开始执行工作流: {workflow_name}")
            
            # 执行每个步骤
            results = {}
            step_params = params.copy()  # 创建参数副本
            
            for i, step in enumerate(steps):
                step_result = self._execute_step(step, step_params, workflow_config)
                results[step] = step_result
                
                # 如果步骤成功，更新下一步的参数
                if step_result.get("success", False):
                    if step == "book_chapter" and "file_path" in step_result:
                        step_params["file_path"] = step_result["file_path"]
                
                # 记录步骤结果
                self.current_workflow["steps"].append({
                    "step": step,
                    "result": step_result,
                    "timestamp": datetime.now()
                })
                
                # 检查步骤是否成功
                if not step_result.get("success", False):
                    error_msg = f"步骤 {step} 执行失败: {step_result.get('error', '未知错误')}"
                    
                    # 重试逻辑
                    if workflow_config.get("retry_on_failure", False):
                        retry_count = 0
                        max_retries = workflow_config.get("max_retries", 1)
                        
                        while retry_count < max_retries:
                            retry_count += 1
                            self._log_workflow("RETRY", f"重试步骤 {step} (第{retry_count}次)")
                            
                            step_result = self._execute_step(step, params, workflow_config)
                            results[step] = step_result
                            
                            if step_result.get("success", False):
                                break
                    
                    # 如果重试后仍然失败
                    if not step_result.get("success", False):
                        self.current_workflow["status"] = "failed"
                        self._log_workflow("ERROR", error_msg)
                        return {"success": False, "error": error_msg, "results": results}
                
                # 质量门控
                if workflow_config.get("quality_gates", False):
                    quality_check = self._quality_gate(step, step_result)
                    if not quality_check["passed"]:
                        self.current_workflow["status"] = "failed"
                        error_msg = f"质量检查失败: {quality_check['reason']}"
                        self._log_workflow("QUALITY_GATE", error_msg)
                        return {"success": False, "error": error_msg, "results": results}
                
                self.current_workflow["current_step"] = i + 1
            
            # 工作流完成
            self.current_workflow["status"] = "completed"
            self.current_workflow["end_time"] = datetime.now()
            self.current_workflow["duration"] = (
                self.current_workflow["end_time"] - self.current_workflow["start_time"]
            ).total_seconds()
            
            success_msg = f"工作流 {workflow_name} 执行成功"
            self._log_workflow("COMPLETE", success_msg)
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "message": success_msg,
                "results": results,
                "duration": self.current_workflow["duration"]
            }
            
        except Exception as e:
            error_msg = f"工作流执行异常: {str(e)}"
            self._log_workflow("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def _execute_step(self, step: str, params: Dict[str, Any], workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个步骤"""
        self._log_workflow("STEP", f"执行步骤: {step}")
        
        try:
            if step == "book_chapter":
                return self._execute_book_chapter(params)
            elif step == "polish":
                return self._execute_polish(params)
            elif step == "template_manager":
                return self._execute_template_management(params)
            else:
                return {"success": False, "error": f"未知步骤: {step}"}
                
        except Exception as e:
            error_msg = f"步骤 {step} 执行异常: {str(e)}"
            self._log_workflow("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def _execute_book_chapter(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行文章创建步骤"""
        required_params = ["topic", "file_type", "filename"]
        for param in required_params:
            if param not in params:
                return {"success": False, "error": f"缺少必要参数: {param}"}
        
        agent = self.agents["book_chapter"]
        return agent.create_article(
            topic=params["topic"],
            file_type=params["file_type"],
            filename=params["filename"],
            author=params.get("author"),
            custom_params=params.get("custom_params")
        )
    
    def _execute_polish(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行文章润色步骤"""
        if "file_path" not in params:
            return {"success": False, "error": "缺少必要参数: file_path"}
        
        agent = self.agents["polish"]
        return agent.polish_article(
            file_path=params["file_path"],
            options=params.get("polish_options")
        )
    
    def _execute_template_management(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行模板管理步骤"""
        operation = params.get("operation", "list")
        agent = self.agents["template_manager"]
        
        if operation == "list":
            return agent.list_templates(params.get("template_type"))
        elif operation == "create":
            return agent.create_template(
                name=params["name"],
                template_type=params["template_type"],
                content=params["content"],
                description=params.get("description", "")
            )
        elif operation == "update":
            return agent.update_template(
                name=params["name"],
                content=params["content"]
            )
        elif operation == "delete":
            return agent.delete_template(params["name"])
        elif operation == "info":
            return agent.get_template_info(params["name"])
        elif operation == "validate":
            return agent.validate_template(params["name"])
        else:
            return {"success": False, "error": f"未知的模板操作: {operation}"}
    
    def _quality_gate(self, step: str, step_result: Dict[str, Any]) -> Dict[str, Any]:
        """质量门控检查"""
        threshold = self.config["quality_threshold"]
        
        if step == "book_chapter":
            # 检查创建的文件是否存在且内容足够
            if "file_path" in step_result:
                file_path = Path(step_result["file_path"])
                if file_path.exists():
                    content_length = file_path.stat().st_size
                    if content_length < threshold["min_content_length"]:
                        return {
                            "passed": False,
                            "reason": f"内容长度不足: {content_length} < {threshold['min_content_length']}"
                        }
                else:
                    return {"passed": False, "reason": "文件不存在"}
        
        elif step == "polish":
            # 检查润色结果中的错误数量
            if "changes" in step_result:
                if len(step_result["changes"]) > threshold["max_errors"]:
                    return {
                        "passed": False,
                        "reason": f"错误数量过多: {len(step_result['changes'])} > {threshold['max_errors']}"
                    }
        
        return {"passed": True}
    
    def _generate_workflow_id(self) -> str:
        """生成工作流ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"workflow_{timestamp}"
    
    def _log_workflow(self, level: str, message: str) -> None:
        """记录工作流日志"""
        if not self.config["logging"]["enabled"]:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "workflow_id": self.current_workflow["id"] if self.current_workflow else None
        }
        
        # 添加到内存中的历史记录
        self.workflow_history.append(log_entry)
        
        # 写入日志文件
        log_file = Path(self.config["logging"]["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"写入日志失败: {e}")
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流状态"""
        if self.current_workflow and self.current_workflow["id"] == workflow_id:
            return {
                "success": True,
                "workflow": self.current_workflow
            }
        
        return {"success": False, "error": f"工作流不存在: {workflow_id}"}
    
    def get_workflow_history(self, limit: int = 10) -> Dict[str, Any]:
        """获取工作流历史"""
        return {
            "success": True,
            "history": self.workflow_history[-limit:],
            "total": len(self.workflow_history)
        }
    
    def create_custom_workflow(self, name: str, steps: List[str], 
                            quality_gates: bool = True, 
                            retry_on_failure: bool = True) -> Dict[str, Any]:
        """创建自定义工作流"""
        try:
            # 验证步骤
            valid_steps = ["book_chapter", "polish", "template_manager"]
            for step in steps:
                if step not in valid_steps:
                    return {"success": False, "error": f"无效步骤: {step}"}
            
            # 创建工作流
            self.config["workflows"][name] = {
                "steps": steps,
                "quality_gates": quality_gates,
                "retry_on_failure": retry_on_failure,
                "max_retries": 3
            }
            
            return {
                "success": True,
                "message": f"成功创建自定义工作流: {name}",
                "workflow_config": self.config["workflows"][name]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# 使用示例
if __name__ == "__main__":
    orchestrator = WorkflowOrchestrator()
    
    # 示例：创建文章工作流
    result = orchestrator.execute_workflow("create_article", {
        "topic": "机器学习基础",
        "file_type": "markdown",
        "filename": "ML-Basics",
        "author": "AI Assistant",
        "polish_options": ["format_normalization", "consistency_check"]
    })
    
    print("工作流执行结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))