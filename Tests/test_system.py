#!/usr/bin/env python3
"""
测试脚本 - 测试AI知识管理系统的各个组件
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# 添加Agents目录到Python路径
sys.path.append(str(Path(__file__).parent.parent / "Agents"))

from book_chapter_agent import BookChapterAgent
from polish_agent import PolishAgent
from template_manager import TemplateManager
from workflow_orchestrator import WorkflowOrchestrator


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        
    def setup(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp(prefix="ai_system_test_")
        print(f"创建临时测试目录: {self.temp_dir}")
        
    def teardown(self):
        """测试后清理"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"清理临时测试目录: {self.temp_dir}")
    
    def run_test(self, test_name, test_func):
        """运行单个测试"""
        try:
            print(f"运行测试: {test_name}")
            result = test_func()
            if result:
                print(f"✓ {test_name} 通过")
                self.test_results.append({"name": test_name, "status": "passed"})
            else:
                print(f"✗ {test_name} 失败")
                self.test_results.append({"name": test_name, "status": "failed"})
        except Exception as e:
            import traceback
            print(f"✗ {test_name} 异常: {e}")
            print(f"异常详情:")
            traceback.print_exc()
            self.test_results.append({"name": test_name, "status": "error", "error": str(e)})
    
    def test_book_chapter_agent(self):
        """测试BookChapterAgent"""
        agent = BookChapterAgent()
        
        # 测试创建LaTeX文章
        result = agent.create_article(
            topic="测试主题",
            file_type="latex",
            filename="test-article",
            author="测试作者"
        )
        
        if not result.get("success"):
            return False
        
        # 检查文件是否创建
        file_path = Path(result["file_path"])
        if not file_path.exists():
            return False
        
        # 检查文件夹是否创建
        folder_path = Path(result["folder_path"])
        if not folder_path.exists():
            return False
        
        return True
    
    def test_polish_agent(self):
        """测试PolishAgent"""
        # 先创建一个测试文件
        test_content = """
# 测试文章

**作者**：测试作者

## 介绍
这是一个测试文章，用于测试润色功能。

## 主要内容
这里有一些  格式问题，需要  修复。

## 结论
测试完成。
"""
        
        test_file = Path(self.temp_dir) / "test_polish.md"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        agent = PolishAgent()
        result = agent.polish_article(
            str(test_file),
            options=["format_normalization", "grammar_check"]
        )
        
        return result.get("success", False)
    
    def test_template_manager(self):
        """测试TemplateManager"""
        manager = TemplateManager()
        
        # 测试列出模板
        result = manager.list_templates()
        if not result.get("success"):
            return False
        
        # 测试获取模板信息
        if "latex-template" in result["templates"]:
            info_result = manager.get_template_info("latex-template")
            if not info_result.get("success"):
                return False
        
        return True
    
    def test_workflow_orchestrator(self):
        """测试WorkflowOrchestrator"""
        orchestrator = WorkflowOrchestrator()
        
        # 测试创建文章工作流
        params = {
            "topic": "工作流测试",
            "file_type": "markdown",
            "filename": "workflow-test",
            "author": "测试作者",
            "polish_options": ["format_normalization"]
        }
        
        result = orchestrator.execute_workflow("create_article", params)
        return result.get("success", False)
    
    def test_end_to_end(self):
        """端到端测试"""
        orchestrator = WorkflowOrchestrator()
        
        # 完整的创建和润色流程
        params = {
            "topic": "端到端测试",
            "file_type": "markdown",
            "filename": "e2e-test",
            "author": "测试作者",
            "polish_options": ["format_normalization", "consistency_check", "grammar_check"]
        }
        
        result = orchestrator.execute_workflow("create_article", params)
        
        if not result.get("success"):
            return False
        
        # 检查生成的文件
        if "book_chapter" in result["results"]:
            book_result = result["results"]["book_chapter"]
            if book_result.get("success"):
                file_path = Path(book_result["file_path"])
                if not file_path.exists():
                    return False
        
        return True
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始运行AI知识管理系统测试...")
        print("=" * 50)
        
        self.setup()
        
        try:
            # 运行各个测试
            self.run_test("BookChapterAgent", self.test_book_chapter_agent)
            self.run_test("PolishAgent", self.test_polish_agent)
            self.run_test("TemplateManager", self.test_template_manager)
            self.run_test("WorkflowOrchestrator", self.test_workflow_orchestrator)
            self.run_test("端到端测试", self.test_end_to_end)
            
        finally:
            self.teardown()
        
        # 输出测试结果
        print("=" * 50)
        print("测试结果汇总:")
        
        passed = sum(1 for r in self.test_results if r["status"] == "passed")
        failed = sum(1 for r in self.test_results if r["status"] == "failed")
        errors = sum(1 for r in self.test_results if r["status"] == "error")
        
        print(f"通过: {passed}")
        print(f"失败: {failed}")
        print(f"错误: {errors}")
        print(f"总计: {len(self.test_results)}")
        
        if failed > 0 or errors > 0:
            print("\n失败的测试:")
            for result in self.test_results:
                if result["status"] != "passed":
                    print(f"  - {result['name']}: {result.get('error', '未知错误')}")
        
        return failed == 0 and errors == 0


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("AI知识管理系统测试脚本")
        print("用法: python test.py [选项]")
        print("选项:")
        print("  --help    显示帮助信息")
        print("  --verbose 详细输出")
        return
    
    runner = TestRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 有测试失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()