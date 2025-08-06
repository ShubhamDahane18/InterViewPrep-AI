# import streamlit as st
# import requests
# import zipfile
# import io
# import os
# import tempfile
# from pathlib import Path
# import json
# from typing import List, Dict, Any
# import base64

# # Configure Streamlit page
# st.set_page_config(
#     page_title="GitHub Code Interview",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# class GitHubCodeAnalyzer:
#     def __init__(self, api_key: str = None):
#         self.api_key = api_key
#         self.supported_extensions = {
#             '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', 
#             '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
#             '.html', '.css', '.sql', '.sh', '.yml', '.yaml', '.json', '.xml',
#             '.md', '.txt', '.dockerfile', '.r', '.m', '.pl'
#         }
        
#     def extract_github_info(self, url: str) -> tuple:
#         """Extract owner and repo name from GitHub URL"""
#         try:
#             if 'github.com' not in url:
#                 return None, None
            
#             parts = url.split('/')
#             owner = parts[-2]
#             repo = parts[-1].replace('.git', '')
#             return owner, repo
#         except:
#             return None, None
    
#     def download_repo(self, owner: str, repo: str) -> str:
#         """Download GitHub repository as ZIP"""
#         download_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"
        
#         try:
#             response = requests.get(download_url, timeout=30)
#             if response.status_code == 404:
#                 # Try master branch if main doesn't exist
#                 download_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
#                 response = requests.get(download_url, timeout=30)
            
#             response.raise_for_status()
#             return response.content
#         except Exception as e:
#             st.error(f"Error downloading repository: {str(e)}")
#             return None
    
#     def extract_code_files(self, zip_content: bytes) -> Dict[str, str]:
#         """Extract and read code files from ZIP"""
#         code_files = {}
        
#         try:
#             with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
#                 for file_info in zip_ref.filelist:
#                     if file_info.is_dir():
#                         continue
                    
#                     file_path = Path(file_info.filename)
#                     if file_path.suffix.lower() in self.supported_extensions:
#                         try:
#                             with zip_ref.open(file_info.filename) as file:
#                                 content = file.read().decode('utf-8', errors='ignore')
#                                 # Remove the root directory from path
#                                 clean_path = '/'.join(file_path.parts[1:]) if len(file_path.parts) > 1 else file_path.name
#                                 code_files[clean_path] = content
#                         except Exception as e:
#                             st.warning(f"Could not read file {file_info.filename}: {str(e)}")
#                             continue
        
#         except Exception as e:
#             st.error(f"Error extracting files: {str(e)}")
        
#         return code_files
    
#     def prepare_context(self, code_files: Dict[str, str], max_context_length: int = 8000) -> str:
#         """Prepare code context for the LLM"""
#         context = "Repository Code Analysis:\n\n"
        
#         # Sort files by importance (main files first)
#         important_files = []
#         other_files = []
        
#         for filepath, content in code_files.items():
#             filename = Path(filepath).name.lower()
#             if any(name in filename for name in ['main', 'index', 'app', 'server', 'client', 'readme']):
#                 important_files.append((filepath, content))
#             else:
#                 other_files.append((filepath, content))
        
#         # Combine files, prioritizing important ones
#         all_files = important_files + other_files
        
#         current_length = 0
#         for filepath, content in all_files:
#             file_section = f"=== {filepath} ===\n{content}\n\n"
#             if current_length + len(file_section) > max_context_length:
#                 # Truncate if too long
#                 remaining_space = max_context_length - current_length - 100
#                 if remaining_space > 0:
#                     context += f"=== {filepath} (truncated) ===\n{content[:remaining_space]}...\n\n"
#                 break
#             context += file_section
#             current_length += len(file_section)
        
#         return context
    
#     def query_llm(self, context: str, question: str) -> str:
#         """Query the LLM with context and question"""
#         if not self.api_key:
#             return "Please provide an API key to use the LLM functionality."
        
#         # Using Groq API (free tier available)
#         url = "https://api.groq.com/openai/v1/chat/completions"
        
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
        
#         prompt = f"""You are a code analysis expert. Analyze the following repository code and answer the user's question.

# Repository Code:
# {context}

# User Question: {question}

# Please provide a detailed, accurate answer based on the code provided. If you need to reference specific files or functions, mention them clearly."""

#         data = {
#             "model": "llama3-8b-8192",  # Free model on Groq
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             "max_tokens": 1000,
#             "temperature": 0.3
#         }
        
#         try:
#             response = requests.post(url, headers=headers, json=data, timeout=30)
#             response.raise_for_status()
            
#             result = response.json()
#             return result['choices'][0]['message']['content']
        
#         except Exception as e:
#             return f"Error querying LLM: {str(e)}\n\nNote: This app uses Groq API. Please ensure you have a valid API key from https://console.groq.com/"

# def main():
#     st.title("🔍 GitHub Repository Code Interview")
#     st.markdown("Upload a GitHub repository URL and ask questions about the code!")
    
#     # Sidebar for configuration
#     with st.sidebar:
#         st.header("Configuration")
        
#         # API Key input
#         api_key = st.text_input(
#             "Groq API Key",
#             type="password",
#             help="Get your free API key from https://console.groq.com/"
#         )
        
#         if st.button("Get Free API Key"):
#             st.markdown("[Click here to get your free Groq API key](https://console.groq.com/)")
        
#         st.markdown("---")
#         st.markdown("### Supported File Types")
#         st.markdown("""
#         - Python (.py)
#         - JavaScript/TypeScript (.js, .jsx, .ts, .tsx)
#         - Java (.java)
#         - C/C++ (.c, .cpp, .h)
#         - C# (.cs)
#         - Go (.go)
#         - Rust (.rs)
#         - And many more...
#         """)
    
#     # Initialize analyzer
#     analyzer = GitHubCodeAnalyzer(api_key)
    
#     # Main interface
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("Repository Input")
        
#         # GitHub URL input
#         github_url = st.text_input(
#             "GitHub Repository URL",
#             placeholder="https://github.com/username/repository",
#             help="Enter the full GitHub repository URL"
#         )
        
#         if st.button("Analyze Repository", type="primary"):
#             if not github_url:
#                 st.error("Please enter a GitHub repository URL")
#                 return
            
#             # Extract repo info
#             owner, repo = analyzer.extract_github_info(github_url)
#             if not owner or not repo:
#                 st.error("Invalid GitHub URL format")
#                 return
            
#             with st.spinner(f"Downloading and analyzing {owner}/{repo}..."):
#                 # Download repository
#                 zip_content = analyzer.download_repo(owner, repo)
#                 if not zip_content:
#                     return
                
#                 # Extract code files
#                 code_files = analyzer.extract_code_files(zip_content)
#                 if not code_files:
#                     st.error("No supported code files found in the repository")
#                     return
                
#                 # Store in session state
#                 st.session_state['code_files'] = code_files
#                 st.session_state['repo_name'] = f"{owner}/{repo}"
#                 st.session_state['context'] = analyzer.prepare_context(code_files)
                
#                 st.success(f"Successfully analyzed {len(code_files)} files from {owner}/{repo}")
    
#     with col2:
#         st.header("Code Files")
        
#         if 'code_files' in st.session_state:
#             st.write(f"**Repository:** {st.session_state['repo_name']}")
#             st.write(f"**Files loaded:** {len(st.session_state['code_files'])}")
            
#             # Show file list
#             with st.expander("View Files", expanded=False):
#                 for filepath in sorted(st.session_state['code_files'].keys()):
#                     st.code(filepath, language=None)
    
#     # Q&A Section
#     st.header("Ask Questions About the Code")
    
#     if 'code_files' in st.session_state:
#         # Sample questions
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             if st.button("What does this project do?"):
#                 st.session_state['current_question'] = "What does this project do? Provide an overview of its main functionality."
        
#         with col2:
#             if st.button("What's the main architecture?"):
#                 st.session_state['current_question'] = "What is the main architecture and structure of this codebase?"
        
#         with col3:
#             if st.button("How do I run this project?"):
#                 st.session_state['current_question'] = "How do I run or deploy this project? What are the setup instructions?"
        
#         # Question input
#         question = st.text_area(
#             "Your Question",
#             value=st.session_state.get('current_question', ''),
#             placeholder="Ask anything about the code: What does this function do? How is the project structured? What are the main features?",
#             height=100
#         )
        
#         if st.button("Get Answer", type="primary"):
#             if not question:
#                 st.error("Please enter a question")
#             elif not api_key:
#                 st.error("Please provide your Groq API key in the sidebar")
#             else:
#                 with st.spinner("Analyzing code and generating answer..."):
#                     answer = analyzer.query_llm(st.session_state['context'], question)
                    
#                     st.header("Answer")
#                     st.markdown(answer)
        
#         # Chat history
#         if 'chat_history' not in st.session_state:
#             st.session_state['chat_history'] = []
        
#         if st.session_state['chat_history']:
#             st.header("Chat History")
#             for i, (q, a) in enumerate(st.session_state['chat_history']):
#                 with st.expander(f"Q{i+1}: {q[:50]}..."):
#                     st.write("**Question:**", q)
#                     st.write("**Answer:**", a)
    
#     else:
#         st.info("Please enter a GitHub repository URL and click 'Analyze Repository' to start asking questions about the code.")
    
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     **How to use:**
#     1. Get a free API key from [Groq Console](https://console.groq.com/)
#     2. Enter your API key in the sidebar
#     3. Paste a GitHub repository URL
#     4. Click 'Analyze Repository'
#     5. Ask questions about the code!
    
#     **Note:** This app uses Groq's free API which provides fast inference for code analysis.
#     """)

# if __name__ == "__main__":
#     main()


import streamlit as st
import requests
import zipfile
import io
import os
import tempfile
from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Tuple
import base64
import time
import asyncio
import concurrent.futures
from dataclasses import dataclass
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import re
import ast
import subprocess
import threading
from queue import Queue
import uuid

# Configure Streamlit page
st.set_page_config(
    page_title="AI Code Analysis Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
}
.agent-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}
.status-running { color: #28a745; }
.status-completed { color: #007bff; }
.status-error { color: #dc3545; }
.code-insight {
    background: #e8f4fd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
}
</style>
""", unsafe_allow_html=True)

@dataclass
class CodeMetrics:
    total_lines: int
    total_files: int
    languages: Dict[str, int]
    complexity_score: float
    test_coverage: float
    documentation_score: float
    security_issues: int
    performance_issues: int

@dataclass
class AgentTask:
    id: str
    name: str
    description: str
    status: str
    result: Optional[str] = None
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class LLMProvider:
    """Base class for LLM providers"""
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        raise NotImplementedError

class GroqProvider(LLMProvider):
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

class OpenAIProvider(LLMProvider):
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            # "model": "gpt-3.5-turbo",
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

class HuggingFaceProvider(LLMProvider):
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"inputs": prompt, "parameters": {"max_length": max_tokens}}
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()[0]['generated_text']
        except Exception as e:
            return f"Error: {str(e)}"

class CodeAnalysisAgent:
    """Specialized agent for different types of code analysis"""
    
    def __init__(self, name: str, llm_provider: LLMProvider, specialty: str):
        self.name = name
        self.llm_provider = llm_provider
        self.specialty = specialty
        self.task_queue = Queue()
        self.results = {}
    
    def analyze_architecture(self, code_files: Dict[str, str]) -> str:
        """Analyze software architecture and design patterns"""
        prompt = f"""As a software architecture expert, analyze this codebase and provide insights on:

1. Overall architecture pattern (MVC, microservices, monolithic, etc.)
2. Design patterns used
3. Code organization and structure
4. Dependencies and coupling
5. Scalability considerations
6. Recommended improvements

Code files:
{self._prepare_code_context(code_files)}

Provide a detailed architectural analysis."""
        
        return self.llm_provider.generate(prompt, max_tokens=1500)
    
    def analyze_security(self, code_files: Dict[str, str]) -> str:
        """Analyze security vulnerabilities"""
        prompt = f"""As a cybersecurity expert, analyze this code for security vulnerabilities:

1. Input validation issues
2. SQL injection vulnerabilities
3. XSS vulnerabilities
4. Authentication/authorization flaws
5. Data exposure risks
6. Dependency vulnerabilities
7. Security best practices violations

Code files:
{self._prepare_code_context(code_files)}

Provide specific security recommendations with code examples."""
        
        return self.llm_provider.generate(prompt, max_tokens=1500)
    
    def analyze_performance(self, code_files: Dict[str, str]) -> str:
        """Analyze performance bottlenecks"""
        prompt = f"""As a performance optimization expert, analyze this code for:

1. Algorithmic complexity issues
2. Memory usage problems
3. Database query optimization
4. Caching opportunities
5. Async/parallel processing opportunities
6. Resource utilization
7. Performance anti-patterns

Code files:
{self._prepare_code_context(code_files)}

Provide specific performance optimization recommendations."""
        
        return self.llm_provider.generate(prompt, max_tokens=1500)
    
    def analyze_testing(self, code_files: Dict[str, str]) -> str:
        """Analyze testing strategy and coverage"""
        prompt = f"""As a testing expert, analyze this codebase for:

1. Current testing approach and coverage
2. Missing test cases
3. Test quality and maintainability
4. Integration testing opportunities
5. Mock/stub usage
6. Test automation possibilities
7. Testing best practices

Code files:
{self._prepare_code_context(code_files)}

Provide testing strategy recommendations and example test cases."""
        
        return self.llm_provider.generate(prompt, max_tokens=1500)
    
    def generate_documentation(self, code_files: Dict[str, str]) -> str:
        """Generate comprehensive documentation"""
        prompt = f"""As a technical documentation expert, create comprehensive documentation for this codebase:

1. Project overview and purpose
2. Installation and setup instructions
3. API documentation
4. Usage examples
5. Configuration options
6. Troubleshooting guide
7. Contributing guidelines

Code files:
{self._prepare_code_context(code_files)}

Generate professional, markdown-formatted documentation."""
        
        return self.llm_provider.generate(prompt, max_tokens=2000)
    
    def suggest_refactoring(self, code_files: Dict[str, str]) -> str:
        """Suggest code refactoring opportunities"""
        prompt = f"""As a code quality expert, analyze this code for refactoring opportunities:

1. Code duplication removal
2. Function/class extraction
3. Variable/method renaming
4. Design pattern applications
5. SOLID principles violations
6. Code smell elimination
7. Maintainability improvements

Code files:
{self._prepare_code_context(code_files)}

Provide specific refactoring suggestions with before/after examples."""
        
        return self.llm_provider.generate(prompt, max_tokens=1500)
    
    def _prepare_code_context(self, code_files: Dict[str, str], max_length: int = 6000) -> str:
        """Prepare code context for analysis"""
        context = ""
        current_length = 0
        
        for filepath, content in code_files.items():
            file_section = f"\n=== {filepath} ===\n{content}\n"
            if current_length + len(file_section) > max_length:
                break
            context += file_section
            current_length += len(file_section)
        
        return context

class AgenticCodeAnalyzer:
    """Main class orchestrating multiple AI agents for comprehensive code analysis"""
    
    def __init__(self):
        self.agents = {}
        self.active_tasks = {}
        self.completed_tasks = {}
        self.code_metrics = None
        
        # Supported file extensions
        self.supported_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', 
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
            '.html', '.css', '.sql', '.sh', '.yml', '.yaml', '.json', '.xml',
            '.md', '.txt', '.dockerfile', '.r', '.m', '.pl', '.vue', '.svelte'
        }
    
    def initialize_agents(self, providers: Dict[str, LLMProvider]):
        """Initialize specialized agents with different LLM providers"""
        agent_configs = [
            ("Architecture Analyst", "architecture", providers.get('primary')),
            ("Security Auditor", "security", providers.get('security')),
            ("Performance Optimizer", "performance", providers.get('performance')),
            ("Test Engineer", "testing", providers.get('testing')),
            ("Documentation Writer", "documentation", providers.get('docs')),
            ("Refactoring Expert", "refactoring", providers.get('refactoring'))
        ]
        
        for name, specialty, provider in agent_configs:
            if provider:
                self.agents[specialty] = CodeAnalysisAgent(name, provider, specialty)
    
    def extract_github_info(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract owner and repo name from GitHub URL"""
        try:
            if 'github.com' not in url:
                return None, None
            
            parts = url.split('/')
            owner = parts[-2]
            repo = parts[-1].replace('.git', '')
            return owner, repo
        except:
            return None, None
    
    def download_repo(self, owner: str, repo: str) -> Optional[bytes]:
        """Download GitHub repository as ZIP"""
        for branch in ['main', 'master', 'develop']:
            download_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
            try:
                response = requests.get(download_url, timeout=30)
                if response.status_code == 200:
                    return response.content
            except:
                continue
        return None
    
    def extract_code_files(self, zip_content: bytes) -> Dict[str, str]:
        """Extract and read code files from ZIP"""
        code_files = {}
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if file_info.is_dir() or file_info.file_size > 1024*1024:  # Skip large files
                        continue
                    
                    file_path = Path(file_info.filename)
                    if file_path.suffix.lower() in self.supported_extensions:
                        try:
                            with zip_ref.open(file_info.filename) as file:
                                content = file.read().decode('utf-8', errors='ignore')
                                clean_path = '/'.join(file_path.parts[1:]) if len(file_path.parts) > 1 else file_path.name
                                code_files[clean_path] = content
                        except:
                            continue
        except Exception as e:
            st.error(f"Error extracting files: {str(e)}")
        
        return code_files
    
    def calculate_metrics(self, code_files: Dict[str, str]) -> CodeMetrics:
        """Calculate comprehensive code metrics"""
        total_lines = 0
        languages = {}
        complexity_score = 0
        
        for filepath, content in code_files.items():
            lines = len(content.split('\n'))
            total_lines += lines
            
            # Language detection
            ext = Path(filepath).suffix.lower()
            lang = self._get_language_from_extension(ext)
            languages[lang] = languages.get(lang, 0) + lines
            
            # Simple complexity calculation
            complexity_score += self._calculate_complexity(content, ext)
        
        # Normalize complexity
        complexity_score = min(complexity_score / len(code_files), 10.0)
        
        # Mock additional metrics (in real implementation, use proper tools)
        test_coverage = self._estimate_test_coverage(code_files)
        documentation_score = self._estimate_documentation_score(code_files)
        security_issues = self._estimate_security_issues(code_files)
        performance_issues = self._estimate_performance_issues(code_files)
        
        return CodeMetrics(
            total_lines=total_lines,
            total_files=len(code_files),
            languages=languages,
            complexity_score=complexity_score,
            test_coverage=test_coverage,
            documentation_score=documentation_score,
            security_issues=security_issues,
            performance_issues=performance_issues
        )
    
    def _get_language_from_extension(self, ext: str) -> str:
        """Map file extension to programming language"""
        lang_map = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.cs': 'C#',
            '.php': 'PHP', '.rb': 'Ruby', '.go': 'Go', '.rs': 'Rust',
            '.html': 'HTML', '.css': 'CSS', '.sql': 'SQL',
            '.sh': 'Shell', '.yml': 'YAML', '.json': 'JSON'
        }
        return lang_map.get(ext, 'Other')
    
    def _calculate_complexity(self, content: str, ext: str) -> float:
        """Simple complexity calculation"""
        # Count control structures
        complexity_keywords = ['if', 'for', 'while', 'switch', 'case', 'try', 'catch']
        complexity = sum(content.lower().count(keyword) for keyword in complexity_keywords)
        return complexity * 0.1
    
    def _estimate_test_coverage(self, code_files: Dict[str, str]) -> float:
        """Estimate test coverage percentage"""
        test_files = [f for f in code_files.keys() if 'test' in f.lower() or 'spec' in f.lower()]
        return min((len(test_files) / len(code_files)) * 100, 100.0)
    
    def _estimate_documentation_score(self, code_files: Dict[str, str]) -> float:
        """Estimate documentation quality score"""
        doc_files = [f for f in code_files.keys() if f.lower().endswith(('.md', '.rst', '.txt'))]
        comment_ratio = 0
        
        for content in code_files.values():
            lines = content.split('\n')
            comment_lines = sum(1 for line in lines if line.strip().startswith(('#', '//', '/*', '*', '"""', "'''")))
            if lines:
                comment_ratio += comment_lines / len(lines)
        
        if code_files:
            comment_ratio /= len(code_files)
        
        return min((len(doc_files) * 20 + comment_ratio * 100), 100.0)
    
    def _estimate_security_issues(self, code_files: Dict[str, str]) -> int:
        """Estimate number of potential security issues"""
        security_patterns = [
            r'eval\s*\(', r'exec\s*\(', r'sql.*=.*\+', r'password\s*=\s*["\']',
            r'api_key\s*=\s*["\']', r'innerHTML\s*=', r'document\.write'
        ]
        
        issues = 0
        for content in code_files.values():
            for pattern in security_patterns:
                issues += len(re.findall(pattern, content, re.IGNORECASE))
        
        return issues
    
    def _estimate_performance_issues(self, code_files: Dict[str, str]) -> int:
        """Estimate number of potential performance issues"""
        performance_patterns = [
            r'for.*in.*:', r'while.*True:', r'\.append\(.*for.*in',
            r'SELECT\s+\*\s+FROM', r'nested.*loop'
        ]
        
        issues = 0
        for content in code_files.values():
            for pattern in performance_patterns:
                issues += len(re.findall(pattern, content, re.IGNORECASE))
        
        return issues
    
    def run_agent_analysis(self, agent_type: str, code_files: Dict[str, str]) -> str:
        """Run analysis with specific agent"""
        if agent_type not in self.agents:
            return "Agent not available"
        
        agent = self.agents[agent_type]
        
        if agent_type == 'architecture':
            return agent.analyze_architecture(code_files)
        elif agent_type == 'security':
            return agent.analyze_security(code_files)
        elif agent_type == 'performance':
            return agent.analyze_performance(code_files)
        elif agent_type == 'testing':
            return agent.analyze_testing(code_files)
        elif agent_type == 'documentation':
            return agent.generate_documentation(code_files)
        elif agent_type == 'refactoring':
            return agent.suggest_refactoring(code_files)
        
        return "Analysis type not supported"

def create_metrics_dashboard(metrics: CodeMetrics):
    """Create interactive metrics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📄 {metrics.total_files}</h3>
            <p>Total Files</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📝 {metrics.total_lines:,}</h3>
            <p>Lines of Code</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 {metrics.test_coverage:.1f}%</h3>
            <p>Test Coverage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⭐ {metrics.complexity_score:.1f}/10</h3>
            <p>Complexity Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Language distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        if metrics.languages:
            fig = px.pie(
                values=list(metrics.languages.values()),
                names=list(metrics.languages.keys()),
                title="Language Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Quality metrics radar chart
        categories = ['Test Coverage', 'Documentation', 'Security', 'Performance', 'Complexity']
        values = [
            metrics.test_coverage,
            metrics.documentation_score,
            max(0, 100 - metrics.security_issues * 10),
            max(0, 100 - metrics.performance_issues * 10),
            max(0, 100 - metrics.complexity_score * 10)
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Code Quality'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Code Quality Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Advanced AI Code Analysis Platform</h1>
        <p>Multi-Agent AI System for Comprehensive Code Analysis, Security Auditing, Performance Optimization & More</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AgenticCodeAnalyzer()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 AI Agent Configuration")
        
        # API Key configuration
        st.subheader("LLM Providers")
        
        groq_key = st.text_input("Groq API Key", type="password", help="Primary LLM for general analysis")
        openai_key = st.text_input("OpenAI API Key", type="password", help="For advanced reasoning tasks")
        hf_key = st.text_input("HuggingFace API Key", type="password", help="For specialized models")
        
        # Agent assignment
        st.subheader("Agent Assignments")
        primary_provider = st.selectbox("Primary Agent", ["Groq", "OpenAI", "HuggingFace"])
        security_provider = st.selectbox("Security Agent", ["Groq", "OpenAI", "HuggingFace"])
        performance_provider = st.selectbox("Performance Agent", ["Groq", "OpenAI", "HuggingFace"])
        
        # Initialize providers
        providers = {}
        if groq_key:
            providers['Groq'] = GroqProvider(groq_key)
        if openai_key:
            providers['OpenAI'] = OpenAIProvider(openai_key)
        if hf_key:
            providers['HuggingFace'] = HuggingFaceProvider(hf_key)
        
        # Map agents to providers
        agent_providers = {
            'primary': providers.get(primary_provider),
            'security': providers.get(security_provider),
            'performance': providers.get(performance_provider),
            'testing': providers.get(primary_provider),
            'docs': providers.get(primary_provider),
            'refactoring': providers.get(primary_provider)
        }
        
        if any(agent_providers.values()):
            st.session_state.analyzer.initialize_agents(agent_providers)
            st.success(f"✅ {len([p for p in agent_providers.values() if p])} agents initialized")
        
        st.markdown("---")
        st.markdown("### 🎯 Analysis Features")
        st.markdown("""
        - **Architecture Analysis**: Design patterns, structure
        - **Security Audit**: Vulnerability detection
        - **Performance Review**: Optimization opportunities  
        - **Test Analysis**: Coverage and quality
        - **Documentation**: Auto-generation
        - **Refactoring**: Code improvement suggestions
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📂 Repository Input")
        
        github_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/username/repository",
            help="Enter any public GitHub repository URL"
        )
        
        # Analysis options
        st.subheader("Analysis Options")
        analysis_types = st.multiselect(
            "Select Analysis Types",
            ["architecture", "security", "performance", "testing", "documentation", "refactoring"],
            default=["architecture", "security"],
            help="Choose which AI agents to deploy"
        )
        
        auto_analyze = st.checkbox("Auto-run all analyses", help="Automatically run all selected analyses")
        
        if st.button("🚀 Start Analysis", type="primary"):
            if not github_url:
                st.error("Please enter a GitHub repository URL")
                return
            
            if not any(agent_providers.values()):
                st.error("Please configure at least one LLM provider")
                return
            
            # Extract repo info
            owner, repo = st.session_state.analyzer.extract_github_info(github_url)
            if not owner or not repo:
                st.error("Invalid GitHub URL format")
                return
            
            with st.spinner(f"🔄 Downloading and analyzing {owner}/{repo}..."):
                # Download repository
                zip_content = st.session_state.analyzer.download_repo(owner, repo)
                if not zip_content:
                    st.error("Could not download repository")
                    return
                
                # Extract code files
                code_files = st.session_state.analyzer.extract_code_files(zip_content)
                if not code_files:
                    st.error("No supported code files found")
                    return
                
                # Calculate metrics
                metrics = st.session_state.analyzer.calculate_metrics(code_files)
                
                # Store in session state
                st.session_state.code_files = code_files
                st.session_state.repo_name = f"{owner}/{repo}"
                st.session_state.metrics = metrics
                
                st.success(f"✅ Successfully loaded {len(code_files)} files from {owner}/{repo}")
                
                # Auto-analyze if enabled
                if auto_analyze:
                    with st.spinner("🤖 Running AI agent analyses..."):
                        for analysis_type in analysis_types:
                            if analysis_type in st.session_state.analyzer.agents:
                                result = st.session_state.analyzer.run_agent_analysis(analysis_type, code_files)
                                st.session_state.analysis_results[analysis_type] = result
                    st.success("🎉 All analyses completed!")
    
    with col2:
        st.header("📊 Repository Overview")
        
        if 'repo_name' in st.session_state:
            st.write(f"**Repository:** {st.session_state.repo_name}")
            st.write(f"**Files loaded:** {len(st.session_state.code_files)} files")
            
            # File explorer
            with st.expander("📁 File Explorer", expanded=False):
                files_df = pd.DataFrame([
                    {"File": f, "Lines": len(content.split('\n')), "Size": f"{len(content)} chars"}
                    for f, content in st.session_state.code_files.items()
                ])
                st.dataframe(files_df, use_container_width=True)
        else:
            st.info("👆 Enter a GitHub repository URL to start analysis")
    
    # Metrics Dashboard
    if 'metrics' in st.session_state:
        st.header("📈 Code Metrics Dashboard")
        create_metrics_dashboard(st.session_state.metrics)
    
    # AI Agent Analysis Results
    if 'code_files' in st.session_state:
        st.header("🤖 AI Agent Analysis")
        
        # Agent control panel
        col1, col2, col3 = st.columns(3)
        
        available_analyses = ["architecture", "security", "performance", "testing", "documentation", "refactoring"]
        
        for i, analysis_type in enumerate(available_analyses):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(f"🔍 {analysis_type.title()}", key=f"btn_{analysis_type}"):
                    if analysis_type in st.session_state.analyzer.agents:
                        with st.spinner(f"Running {analysis_type} analysis..."):
                            result = st.session_state.analyzer.run_agent_analysis(
                                analysis_type, st.session_state.code_files
                            )
                            st.session_state.analysis_results[analysis_type] = result
                    else:
                        st.error(f"No agent configured for {analysis_type}")
        
        # Display analysis results
        if st.session_state.analysis_results:
            st.subheader("🔬 Analysis Results")
            
            for analysis_type, result in st.session_state.analysis_results.items():
                with st.expander(f"📋 {analysis_type.title()} Analysis", expanded=True):
                    st.markdown(f"""
                    <div class="code-insight">
                        <h4>🤖 {analysis_type.title()} Agent Report</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(result)
        
        # Interactive Q&A
        st.header("💬 Interactive Code Q&A")
        
        question = st.text_area(
            "Ask questions about your code",
            placeholder="Example: How can I improve the security of this authentication system?",
            height=100
        )
        
        if st.button("🎯 Get AI Answer") and question:
            if st.session_state.analyzer.agents:
                # Use primary agent for Q&A
                primary_agent = next(iter(st.session_state.analyzer.agents.values()))
                with st.spinner("🤔 AI is analyzing your question..."):
                    context = primary_agent._prepare_code_context(st.session_state.code_files)
                    qa_prompt = f"""You are an expert code analyst. Based on the following codebase, answer the user's question comprehensively.

Codebase:
{context}

User Question: {question}

Provide a detailed, technical answer with specific code examples and recommendations."""
                    
                    answer = primary_agent.llm_provider.generate(qa_prompt, max_tokens=1500)
                    
                    # Store in chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append((question, answer))
                    
                    st.markdown("### 🤖 AI Response")
                    st.markdown(answer)
        
        # Chat History
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            st.header("💭 Chat History")
            for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {q[:60]}..."):
                    st.markdown(f"**Question:** {q}")
                    st.markdown(f"**Answer:** {a}")
    
    # Advanced Features Section
    st.header("🚀 Advanced Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h4>🔄 Continuous Monitoring</h4>
            <p>Set up automated code quality monitoring with webhooks and notifications.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Setup Monitoring"):
            st.info("🚧 Monitoring feature coming soon! Will include GitHub webhook integration.")
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <h4>📊 Comparative Analysis</h4>
            <p>Compare multiple repositories or track changes over time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Compare Repos"):
            st.info("🚧 Repository comparison feature in development!")
    
    with col3:
        st.markdown("""
        <div class="agent-card">
            <h4>🎯 Custom Agents</h4>
            <p>Create specialized AI agents for your specific use cases.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Create Agent"):
            st.info("🚧 Custom agent builder coming soon!")
    
    # Export and Reporting
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        st.header("📤 Export & Reporting")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate Report"):
                # Create comprehensive report
                report = f"""# Code Analysis Report
Repository: {st.session_state.get('repo_name', 'Unknown')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Metrics Summary
- Total Files: {st.session_state.metrics.total_files}
- Lines of Code: {st.session_state.metrics.total_lines:,}
- Test Coverage: {st.session_state.metrics.test_coverage:.1f}%
- Complexity Score: {st.session_state.metrics.complexity_score:.1f}/10

## AI Agent Analysis Results

"""
                for analysis_type, result in st.session_state.analysis_results.items():
                    report += f"### {analysis_type.title()} Analysis\n{result}\n\n"
                
                st.download_button(
                    "💾 Download Report",
                    report,
                    file_name=f"code_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col2:
            if st.button("📊 Export Metrics"):
                metrics_data = {
                    "repository": st.session_state.get('repo_name', 'Unknown'),
                    "total_files": st.session_state.metrics.total_files,
                    "total_lines": st.session_state.metrics.total_lines,
                    "test_coverage": st.session_state.metrics.test_coverage,
                    "complexity_score": st.session_state.metrics.complexity_score,
                    "languages": st.session_state.metrics.languages,
                    "timestamp": datetime.now().isoformat()
                }
                
                st.download_button(
                    "💾 Download Metrics JSON",
                    json.dumps(metrics_data, indent=2),
                    file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📧 Share Results"):
                st.info("🚧 Email sharing feature coming soon!")
    
    # Real-time Agent Status (Mock)
    if st.session_state.analyzer.agents:
        st.header("⚡ Agent Status Monitor")
        
        status_cols = st.columns(len(st.session_state.analyzer.agents))
        
        for i, (agent_type, agent) in enumerate(st.session_state.analyzer.agents.items()):
            with status_cols[i]:
                # Mock status indicators
                status = "🟢 Active" if agent_type in st.session_state.analysis_results else "⚪ Idle"
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <h5>{agent.name}</h5>
                    <p>{status}</p>
                    <small>Specialty: {agent.specialty}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer with advanced info
    st.markdown("---")
    st.markdown("""
    ## 🎯 Platform Capabilities
    
    **Multi-Agent AI System:**
    - 🏗️ **Architecture Agent**: Analyzes design patterns, structure, and scalability
    - 🔒 **Security Agent**: Identifies vulnerabilities and security best practices
    - ⚡ **Performance Agent**: Finds bottlenecks and optimization opportunities
    - 🧪 **Testing Agent**: Evaluates test coverage and suggests improvements
    - 📚 **Documentation Agent**: Generates comprehensive documentation
    - 🔄 **Refactoring Agent**: Suggests code improvements and clean-up
    
    **Advanced Features:**
    - Multiple LLM provider support (Groq, OpenAI, HuggingFace)
    - Real-time code metrics and quality scoring
    - Interactive Q&A with context-aware responses
    - Comprehensive reporting and export capabilities
    - Visual dashboards and analytics
    
    **Coming Soon:**
    - GitHub webhook integration for continuous monitoring
    - Custom agent creation and training
    - Repository comparison and diff analysis
    - Team collaboration features
    - API access for programmatic integration
    
    ---
    
    **Setup Instructions:**
    1. Get API keys from your preferred LLM providers:
       - [Groq Console](https://console.groq.com/) (Free tier available)
       - [OpenAI API](https://platform.openai.com/) (Pay-per-use)
       - [HuggingFace](https://huggingface.co/settings/tokens) (Free tier available)
    2. Configure agents in the sidebar
    3. Enter a GitHub repository URL
    4. Select analysis types and run comprehensive analysis
    5. Interact with AI agents through Q&A
    6. Export reports and metrics for your team
    
    **Pro Tips:**
    - Use different LLM providers for different agents to get diverse perspectives
    - Enable auto-analysis for quick comprehensive reviews  
    - Ask specific questions about security, performance, or architecture
    - Download reports for documentation and team reviews
    - Monitor code quality metrics over time
    """)

if __name__ == "__main__":
    main()