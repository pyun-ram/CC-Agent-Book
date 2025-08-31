#!/usr/bin/env python3
"""
主系统入口脚本
使用方法: python main.py [command] [options]
"""

import sys
import json
import argparse
from pathlib import Path

# 添加Agents目录到Python路径
sys.path.append(str(Path(__file__).parent / "Agents"))

from workflow_orchestrator import WorkflowOrchestrator


def main():
    parser = argparse.ArgumentParser(description="AI知识管理系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 创建文章命令
    create_parser = subparsers.add_parser("create", help="创建新文章")
    create_parser.add_argument("topic", help="文章主题")
    create_parser.add_argument("file_type", choices=["latex", "markdown"], help="文件类型")
    create_parser.add_argument("filename", help="文件名")
    create_parser.add_argument("--author", help="作者", default="AI System")
    create_parser.add_argument("--polish", action="store_true", help="创建后自动润色")
    
    # 润色文章命令
    polish_parser = subparsers.add_parser("polish", help="润色文章")
    polish_parser.add_argument("file_path", help="文章文件路径")
    polish_parser.add_argument("--options", nargs="+", 
                              choices=["format_normalization", "consistency_check", 
                                     "grammar_check", "academic_polish"],
                              help="润色选项")
    
    # 模板管理命令
    template_parser = subparsers.add_parser("template", help="模板管理")
    template_subparsers = template_parser.add_subparsers(dest="template_command")
    
    # 列出模板
    template_subparsers.add_parser("list", help="列出所有模板")
    
    # 创建模板
    template_create_parser = template_subparsers.add_parser("create", help="创建新模板")
    template_create_parser.add_argument("name", help="模板名称")
    template_create_parser.add_argument("type", choices=["latex", "markdown"], help="模板类型")
    template_create_parser.add_argument("content", help="模板内容")
    template_create_parser.add_argument("--description", help="模板描述")
    
    # 工作流状态命令
    status_parser = subparsers.add_parser("status", help="查看工作流状态")
    status_parser.add_argument("--workflow-id", help="工作流ID")
    
    # 历史记录命令
    history_parser = subparsers.add_parser("history", help="查看工作流历史")
    history_parser.add_argument("--limit", type=int, default=10, help="显示记录数量")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        orchestrator = WorkflowOrchestrator()
        
        if args.command == "create":
            # 创建文章工作流
            params = {
                "topic": args.topic,
                "file_type": args.file_type,
                "filename": args.filename,
                "author": args.author
            }
            
            if args.polish:
                params["polish_options"] = ["format_normalization", "consistency_check"]
                result = orchestrator.execute_workflow("create_article", params)
            else:
                # 直接执行book_chapter步骤
                result = orchestrator._execute_book_chapter(params)
            
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        elif args.command == "polish":
            # 润色文章
            params = {
                "file_path": args.file_path,
                "polish_options": args.options or ["format_normalization", "consistency_check"]
            }
            result = orchestrator.execute_workflow("polish_only", params)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        elif args.command == "template":
            if args.template_command == "list":
                result = orchestrator._execute_template_management({"operation": "list"})
                print(json.dumps(result, ensure_ascii=False, indent=2))
            
            elif args.template_command == "create":
                params = {
                    "operation": "create",
                    "name": args.name,
                    "template_type": args.type,
                    "content": args.content,
                    "description": args.description or ""
                }
                result = orchestrator._execute_template_management(params)
                print(json.dumps(result, ensure_ascii=False, indent=2))
        
        elif args.command == "status":
            if args.workflow_id:
                result = orchestrator.get_workflow_status(args.workflow_id)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("请指定工作流ID")
        
        elif args.command == "history":
            result = orchestrator.get_workflow_history(args.limit)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()