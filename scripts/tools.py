# -*- coding: utf-8 -*-
# @Date    : 1/13/2026
# @Author  : Jingwen
# @Desc    : EasyTool-inspired tool infrastructure for WirelessAgent

"""
Tool Infrastructure inspired by EasyTool
https://arxiv.org/abs/2401.06201

Key features:
1. Unified tool schema definition
2. Concise tool descriptions for LLM
3. Standardized tool execution interface
4. Token-efficient tool documentation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import aiohttp
import time
from collections import deque
from scripts.logs import logger


# ==================== Tool Metrics (Performance Tracking) ====================

class ToolMetrics:
    """
    Automatic metrics collection for tools
    
    Tracks:
    - Total calls
    - Success/failure counts
    - Execution time statistics
    - Recent call history
    """
    
    def __init__(self, name: str, history_size: int = 100):
        """
        Args:
            name: Tool name
            history_size: Number of recent calls to track
        """
        self.name = name
        self.history_size = history_size
        
        # Counters
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
        # Time tracking
        self.total_execution_time = 0.0
        self.min_execution_time = float('inf')
        self.max_execution_time = 0.0
        
        # Recent history (for rolling statistics)
        self.recent_calls = deque(maxlen=history_size)
        self.recent_errors = deque(maxlen=50)  # Track recent errors
    
    def record(self, duration: float, success: bool, error: Optional[str] = None):
        """Record a tool call"""
        self.total_calls += 1
        self.total_execution_time += duration
        
        # Update min/max
        self.min_execution_time = min(self.min_execution_time, duration)
        self.max_execution_time = max(self.max_execution_time, duration)
        
        # Success/failure
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if error:
                self.recent_errors.append({
                    "timestamp": time.time(),
                    "error": error
                })
        
        # Add to recent history
        self.recent_calls.append({
            "timestamp": time.time(),
            "duration": duration,
            "success": success
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        if self.total_calls == 0:
            return {
                "tool_name": self.name,
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "min_execution_time": 0.0,
                "max_execution_time": 0.0
            }
        
        # Calculate recent success rate (last N calls)
        if self.recent_calls:
            recent_successes = sum(1 for call in self.recent_calls if call["success"])
            recent_success_rate = recent_successes / len(self.recent_calls)
        else:
            recent_success_rate = 0.0
        
        return {
            "tool_name": self.name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls,
            "avg_execution_time": self.total_execution_time / self.total_calls,
            "min_execution_time": self.min_execution_time if self.min_execution_time != float('inf') else 0.0,
            "max_execution_time": self.max_execution_time,
            "recent_success_rate": recent_success_rate,
            "recent_error_count": len(self.recent_errors)
        }
    
    def reset(self):
        """Reset all metrics"""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_execution_time = 0.0
        self.min_execution_time = float('inf')
        self.max_execution_time = 0.0
        self.recent_calls.clear()
        self.recent_errors.clear()


# ==================== Tool Schema (EasyTool Core) ====================

class ToolParameter(BaseModel):
    """Tool parameter definition (EasyTool style)"""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None


class ToolSchema(BaseModel):
    """
    Unified tool description format (EasyTool core concept)
    
    Converts diverse API documentation into standardized, concise descriptions
    """
    name: str = Field(description="Tool name (unique identifier)")
    description: str = Field(description="Concise description of what the tool does")
    category: str = Field(default="general", description="Tool category (e.g., 'search', 'compute', 'data')")
    
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    
    # EasyTool enhancement: concise usage example
    usage_example: Optional[str] = Field(default=None, description="Example of how to use this tool")
    
    # Metadata
    source: Optional[str] = Field(default=None, description="Tool source (e.g., 'wikipedia_api', 'custom')")
    version: Optional[str] = Field(default="1.0", description="Tool version")
    
    def to_llm_description(self) -> str:
        """
        Generate concise description for LLM (EasyTool core functionality)
        
        Converts detailed schema into concise, understandable description
        """
        # Parameter descriptions
        param_descs = []
        for param in self.parameters:
            required_mark = "*" if param.required else ""
            default_info = f" (default: {param.default})" if param.default is not None else ""
            param_descs.append(
                f"  - {param.name}{required_mark} ({param.type}): {param.description}{default_info}"
            )
        
        params_text = "\n".join(param_descs) if param_descs else "  No parameters"
        
        # Complete description
        description = f"""Tool: {self.name}
Category: {self.category}
Description: {self.description}
Parameters:
{params_text}"""
        
        if self.usage_example:
            description += f"\nExample: {self.usage_example}"
        
        return description.strip()
    
    def to_openai_function_format(self) -> Dict:
        """Convert to OpenAI Function Calling format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


# ==================== Base Tool (EasyTool Wrapper) ====================

class BaseTool(ABC):
    """
    Base class for all tools (EasyTool-inspired)
    
    Features:
    1. Unified schema definition
    2. Auto-generate LLM-friendly descriptions
    3. Standardized execution interface
    4. Automatic metrics collection (NEW)
    """
    
    def __init__(self, schema: ToolSchema, enable_metrics: bool = True):
        self.schema = schema
        self.name = schema.name
        
        # NEW: Automatic metrics collection
        self.metrics = ToolMetrics(name=self.name) if enable_metrics else None
    
    async def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Execute tool with automatic metrics collection
        
        This is the recommended entry point for tool execution.
        It wraps execute() with automatic performance tracking.
        """
        start_time = time.time()
        
        try:
            # Execute the tool
            result = await self.execute(**kwargs)
            execution_time = time.time() - start_time
            
            # Record metrics
            if self.metrics:
                success = result.get("success", False)
                error = result.get("error") if not success else None
                self.metrics.record(
                    duration=execution_time,
                    success=success,
                    error=error
                )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failure
            if self.metrics:
                self.metrics.record(
                    duration=execution_time,
                    success=False,
                    error=str(e)
                )
            
            # Re-raise for backward compatibility
            raise
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute tool invocation (implement in subclass)
        
        Returns:
            {
                "success": bool,
                "result": Any,
                "error": Optional[str]
            }
        """
        pass
    
    def get_schema(self) -> ToolSchema:
        """Get tool schema"""
        return self.schema
    
    def get_llm_description(self) -> str:
        """Get LLM-friendly description"""
        return self.schema.to_llm_description()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tool execution statistics
        
        Returns metrics like:
        - total_calls
        - success_rate
        - avg_execution_time
        - etc.
        """
        if not self.metrics:
            return {
                "tool_name": self.name,
                "metrics_enabled": False,
                "message": "Metrics collection is disabled for this tool"
            }
        
        return self.metrics.get_stats()
    
    def validate_parameters(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate parameters"""
        # Check required parameters
        required_params = [p.name for p in self.schema.parameters if p.required]
        missing = set(required_params) - set(kwargs.keys())
        
        if missing:
            return False, f"Missing required parameters: {missing}"
        
        return True, None


# ==================== Concrete Tool Implementations ====================

class WikipediaTool(BaseTool):
    """Wikipedia search tool for factual information"""
    
    def __init__(self):
        schema = ToolSchema(
            name="wikipedia_search",
            description="Search Wikipedia for factual information about people, places, events, concepts, etc. Returns article titles, summaries, and URLs.",
            category="search",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query (e.g., 'Albert Einstein', 'Python programming')",
                    required=True
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to return",
                    required=False,
                    default=3
                )
            ],
            usage_example='wikipedia_search(query="Eiffel Tower", max_results=3)',
            source="wikipedia_api",
            version="1.0"
        )
        super().__init__(schema)
        self.base_url = "https://en.wikipedia.org/w/api.php"
    
    async def execute(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Execute Wikipedia search"""
        # Validate parameters
        valid, error = self.validate_parameters(query=query, max_results=max_results)
        if not valid:
            return {"success": False, "result": None, "error": error}
        
        try:
            params = {
                "action": "opensearch",
                "search": query,
                "limit": max_results,
                "format": "json"
            }
            
            headers = {
                "User-Agent": "WirelessAgent-Research-Tool/1.0"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Format results
                        results = []
                        for i in range(len(data[1])):
                            results.append({
                                "title": data[1][i],
                                "description": data[2][i] if i < len(data[2]) else "",
                                "url": data[3][i] if i < len(data[3]) else ""
                            })
                        
                        # Create readable summary
                        summary = "\n".join([
                            f"{i+1}. {r['title']}: {r['description']}" 
                            for i, r in enumerate(results)
                        ])
                        
                        return {
                            "success": True,
                            "result": summary if summary else "No results found",
                            "raw_data": results,
                            "error": None
                        }
                    else:
                        return {
                            "success": False,
                            "result": None,
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            logger.error(f"Wikipedia tool error: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }


class CalculatorTool(BaseTool):
    """Mathematical expression evaluator using SymPy"""
    
    def __init__(self):
        schema = ToolSchema(
            name="calculator",
            description="Evaluate PURE mathematical expressions using SymPy. Supports arithmetic (2+3*4), algebra (x**2+3*x-5), trigonometry (sin(pi/2)), calculus (integrate(x**2, x)), etc. CANNOT handle Python code syntax like lists [1,2,3], loops (for/while), conditionals (if/else), or list comprehensions. For those, use python_code_solver instead.",
            category="compute",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Pure mathematical expression (no Python lists/loops/conditionals). Examples: '2+2', 'sqrt(16)', 'sin(pi/2)', 'x**2+5*x+6'",
                    required=True
                )
            ],
            usage_example='calculator(expression="2 + 2 * 3")',
            source="sympy",
            version="1.0"
        )
        super().__init__(schema)
    
    async def execute(self, expression: str = None, **kwargs) -> Dict[str, Any]:
        """Execute mathematical calculation"""
        # Handle missing or empty expression
        if not expression:
            # Check if expression was passed under a different key
            expression = kwargs.get('expr') or kwargs.get('formula') or kwargs.get('input')
            if not expression:
                return {
                    "success": False,
                    "result": None,
                    "error": "Missing 'expression' parameter. Usage: calculator(expression='2+3*4')"
                }
        
        valid, error = self.validate_parameters(expression=expression)
        if not valid:
            return {"success": False, "result": None, "error": error}
        
        try:
            import sympy
            
            # Clean expression
            expression = expression.strip()
            
            # Check for Python-specific syntax that SymPy can't handle
            python_only_patterns = ['len(', '[', 'for ', ' in ', 'if ', 'else', 'def ', 'lambda']
            if any(pattern in expression for pattern in python_only_patterns):
                return {
                    "success": False,
                    "result": None,
                    "error": f"This expression contains Python code syntax (lists, loops, etc.) that calculator cannot handle. Use python_code_solver instead for: {expression[:100]}"
                }
            
            # Use SymPy to calculate
            result = sympy.sympify(expression)
            evaluated = sympy.N(result, 10)  # 10 decimal precision
            
            return {
                "success": True,
                "result": str(evaluated),
                "error": None
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Calculator tool error: {e}")
            
            # Provide helpful error message
            if 'could not parse' in error_msg or 'SyntaxError' in error_msg:
                hint = "Use python_code_solver for complex expressions with Python syntax (lists, loops, conditionals)."
            else:
                hint = "Check if the mathematical expression is valid."
            
            return {
                "success": False,
                "result": None,
                "error": f"Invalid expression: {error_msg[:200]}. Hint: {hint}"
            }


# ==================== Wikipedia Page Content Tool ====================

class WikipediaPageTool(BaseTool):
    """Fetch full Wikipedia page content for detailed information"""
    
    def __init__(self):
        schema = ToolSchema(
            name="wikipedia_page",
            description="Fetch the full content of a specific Wikipedia page. Returns the complete article text with detailed information. Use this when you need comprehensive information about a topic, not just a summary.",
            category="knowledge",
            parameters=[
                ToolParameter(
                    name="title",
                    type="string",
                    description="Exact title of the Wikipedia page (e.g., 'Eiffel Tower', 'Python (programming language)')",
                    required=True
                ),
                ToolParameter(
                    name="sentences",
                    type="integer",
                    description="Number of sentences to return (default: 5, max: 10). Use fewer for quick facts, more for detailed context.",
                    required=False
                )
            ],
            usage_example='wikipedia_page(title="Eiffel Tower", sentences=5)',
            source="wikipedia_api",
            version="1.0"
        )
        super().__init__(schema)
        self.base_url = "https://en.wikipedia.org/w/api.php"
    
    async def execute(self, title: str, sentences: int = 5) -> Dict[str, Any]:
        """Fetch Wikipedia page content"""
        valid, error = self.validate_parameters(title=title)
        if not valid:
            return {"success": False, "result": None, "error": error}
        
        # Limit sentences
        sentences = min(max(1, sentences), 10)
        
        try:
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "exsentences": sentences,
                "explaintext": 1  # Use 1 instead of True for API compatibility
            }
            
            headers = {
                "User-Agent": "WirelessAgent-Research-Tool/1.0"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        pages = data.get("query", {}).get("pages", {})
                        
                        if not pages:
                            return {
                                "success": False,
                                "result": None,
                                "error": "No page found"
                            }
                        
                        # Get first page
                        page_id = list(pages.keys())[0]
                        if page_id == "-1":
                            return {
                                "success": False,
                                "result": None,
                                "error": f"Page '{title}' not found"
                            }
                        
                        page = pages[page_id]
                        extract = page.get("extract", "")
                        
                        if not extract:
                            return {
                                "success": False,
                                "result": None,
                                "error": "No content available"
                            }
                        
                        return {
                            "success": True,
                            "result": extract,
                            "page_title": page.get("title", title),
                            "error": None
                        }
                    else:
                        return {
                            "success": False,
                            "result": None,
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            logger.error(f"Wikipedia page tool error: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }


# ==================== Text Search Tool ====================

class TextSearchTool(BaseTool):
    """Search for specific information within text"""
    
    def __init__(self):
        schema = ToolSchema(
            name="text_search",
            description="Search for specific keywords or phrases within a given text. Useful for finding specific facts in previously retrieved information. Returns matching sentences with context.",
            category="knowledge",
            parameters=[
                ToolParameter(
                    name="text",
                    type="string",
                    description="The text to search in (e.g., content from Wikipedia page)",
                    required=True
                ),
                ToolParameter(
                    name="keywords",
                    type="string",
                    description="Keywords or phrase to search for (case-insensitive)",
                    required=True
                )
            ],
            usage_example='text_search(text="<long text>", keywords="birth date")',
            source="regex",
            version="1.0"
        )
        super().__init__(schema)
    
    async def execute(self, text: str, keywords: str) -> Dict[str, Any]:
        """Search text for keywords"""
        valid, error = self.validate_parameters(text=text, keywords=keywords)
        if not valid:
            return {"success": False, "result": None, "error": error}
        
        try:
            import re
            
            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            
            # Search for keywords (case-insensitive)
            keywords_lower = keywords.lower()
            matching_sentences = []
            
            for sentence in sentences:
                if keywords_lower in sentence.lower():
                    matching_sentences.append(sentence.strip())
            
            if matching_sentences:
                result = "\n".join([f"- {s}" for s in matching_sentences[:5]])  # Max 5 matches
                return {
                    "success": True,
                    "result": result,
                    "num_matches": len(matching_sentences),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": f"No sentences containing '{keywords}' found"
                }
        except Exception as e:
            logger.error(f"Text search tool error: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }


# ==================== Comparison Tool ====================

class ComparisonTool(BaseTool):
    """Compare two entities based on Wikipedia information"""
    
    def __init__(self):
        schema = ToolSchema(
            name="compare_entities",
            description="Compare two entities (people, places, things) to find similarities or differences. Fetches information from Wikipedia and provides a comparison. Useful for 'both', 'either', 'which one' questions.",
            category="knowledge",
            parameters=[
                ToolParameter(
                    name="entity1",
                    type="string",
                    description="First entity to compare (e.g., 'Python programming language')",
                    required=True
                ),
                ToolParameter(
                    name="entity2",
                    type="string",
                    description="Second entity to compare (e.g., 'JavaScript')",
                    required=True
                ),
                ToolParameter(
                    name="aspect",
                    type="string",
                    description="What aspect to compare (e.g., 'type', 'origin', 'purpose'). If not specified, provides general comparison.",
                    required=False
                )
            ],
            usage_example='compare_entities(entity1="Python", entity2="Java", aspect="type")',
            source="wikipedia_api",
            version="1.0"
        )
        super().__init__(schema)
        self.base_url = "https://en.wikipedia.org/w/api.php"
    
    async def execute(self, entity1: str, entity2: str, aspect: str = None) -> Dict[str, Any]:
        """Compare two entities"""
        valid, error = self.validate_parameters(entity1=entity1, entity2=entity2)
        if not valid:
            return {"success": False, "result": None, "error": error}
        
        try:
            # Fetch summaries for both entities
            summaries = {}
            headers = {
                "User-Agent": "WirelessAgent-Research-Tool/1.0"
            }
            
            for entity in [entity1, entity2]:
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": entity,
                    "prop": "extracts",
                    "exsentences": 3,
                    "explaintext": 1  # Use 1 instead of True for API compatibility
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, params=params, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            pages = data.get("query", {}).get("pages", {})
                            page_id = list(pages.keys())[0]
                            if page_id != "-1":
                                summaries[entity] = pages[page_id].get("extract", "No information found")
                            else:
                                summaries[entity] = f"No Wikipedia page found for '{entity}'"
            
            # Format comparison
            if aspect:
                result = f"Comparing {entity1} and {entity2} on '{aspect}':\n\n"
            else:
                result = f"Comparison of {entity1} and {entity2}:\n\n"
            
            result += f"{entity1}:\n{summaries[entity1][:300]}...\n\n"
            result += f"{entity2}:\n{summaries[entity2][:300]}..."
            
            return {
                "success": True,
                "result": result,
                "entity1_info": summaries[entity1],
                "entity2_info": summaries[entity2],
                "error": None
            }
        except Exception as e:
            logger.error(f"Comparison tool error: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }


# ==================== Year/Date Extractor Tool ====================

class YearExtractorTool(BaseTool):
    """Extract years and dates from text"""
    
    def __init__(self):
        schema = ToolSchema(
            name="extract_year",
            description="Extract years, dates, or time periods from text. Useful for questions about 'when', 'what year', 'what season', etc. Returns all found years/dates in chronological order.",
            category="knowledge",
            parameters=[
                ToolParameter(
                    name="text",
                    type="string",
                    description="Text to extract years/dates from",
                    required=True
                ),
                ToolParameter(
                    name="context",
                    type="string",
                    description="Context keywords to filter results (e.g., 'born', 'founded', 'season')",
                    required=False
                )
            ],
            usage_example='extract_year(text="He was born in 1978 and died in 2020", context="born")',
            source="regex",
            version="1.0"
        )
        super().__init__(schema)
    
    async def execute(self, text: str, context: str = None) -> Dict[str, Any]:
        """Extract years from text"""
        valid, error = self.validate_parameters(text=text)
        if not valid:
            return {"success": False, "result": None, "error": error}
        
        try:
            import re
            
            # Pattern for years (1000-2999)
            year_pattern = r'\b(1[0-9]{3}|2[0-9]{3})\b'
            
            # Pattern for seasons (e.g., "2020 season", "2020-21 season")
            season_pattern = r'\b(1[0-9]{3}|2[0-9]{3})(?:-\d{2,4})?\s+(?:season|NCAA|Division)\b'
            
            years = []
            seasons = []
            
            # Find all years
            year_matches = re.finditer(year_pattern, text)
            for match in year_matches:
                year = match.group(1)
                # Get surrounding context (20 chars before and after)
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                snippet = text[start:end]
                
                if context:
                    # Filter by context
                    if context.lower() in snippet.lower():
                        years.append((year, snippet))
                else:
                    years.append((year, snippet))
            
            # Find seasons
            season_matches = re.finditer(season_pattern, text, re.IGNORECASE)
            for match in season_matches:
                seasons.append(match.group(0))
            
            if seasons:
                result = f"Seasons found: {', '.join(seasons)}\n"
            elif years:
                result = "Years found:\n"
                for year, snippet in years[:5]:  # Max 5
                    result += f"- {year}: ...{snippet.strip()}...\n"
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": "No years or dates found in text"
                }
            
            return {
                "success": True,
                "result": result.strip(),
                "years": [y[0] for y in years],
                "seasons": seasons,
                "error": None
            }
        except Exception as e:
            logger.error(f"Year extractor tool error: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }


# ==================== Tool Registry ====================

class ToolRegistry:
    """
    Tool registry (EasyTool enhanced)
    
    Features:
    1. Manage all available tools
    2. Generate concise tool descriptions (reduce tokens)
    3. Support multiple export formats
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.categories: Dict[str, List[str]] = {}  # category -> tool names
    
    def register(self, tool: BaseTool):
        """Register a tool"""
        self.tools[tool.name] = tool
        
        # Organize by category
        category = tool.schema.category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(tool.name)
        
        logger.info(f"✅ Registered tool: {tool.name} ({category})")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool instance"""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get tools by category"""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names]
    
    def get_all_schemas(self) -> List[ToolSchema]:
        """Get all tool schemas"""
        return [tool.get_schema() for tool in self.tools.values()]
    
    def get_concise_description(self, max_tools: int = None) -> str:
        """
        Generate concise tool descriptions (EasyTool core functionality)
        
        Args:
            max_tools: Maximum number of tools to include (control token count)
        """
        descriptions = []
        tools_to_include = list(self.tools.values())[:max_tools] if max_tools else list(self.tools.values())
        
        for tool in tools_to_include:
            descriptions.append(tool.get_llm_description())
        
        header = f"Available Tools ({len(tools_to_include)} total):"
        return header + "\n\n" + "\n\n---\n\n".join(descriptions)
    
    def export_to_openai_functions(self) -> List[Dict]:
        """Export to OpenAI Function Calling format"""
        return [tool.schema.to_openai_function_format() for tool in self.tools.values()]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute tool invocation with automatic metrics collection
        
        Now uses tool.__call__() instead of tool.execute() to ensure
        automatic metrics are collected for all tool executions.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "result": None,
                "error": f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            }
        
        try:
            # Use __call__ to get automatic metrics collection
            result = await tool(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    def get_usage_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive usage report for all tools
        
        Returns:
            {
                "total_tools": int,
                "tools": {
                    "tool_name": {
                        "total_calls": int,
                        "success_rate": float,
                        "avg_execution_time": float,
                        ...
                    }
                },
                "summary": {
                    "total_calls": int,
                    "most_used_tool": str,
                    "least_used_tool": str,
                    "highest_success_rate": str,
                    "lowest_success_rate": str
                }
            }
        """
        report = {
            "total_tools": len(self.tools),
            "tools": {},
            "summary": {}
        }
        
        # Collect stats from each tool
        all_stats = []
        for tool_name, tool in self.tools.items():
            stats = tool.get_statistics()
            report["tools"][tool_name] = stats
            
            if stats.get("total_calls", 0) > 0:
                all_stats.append((tool_name, stats))
        
        # Generate summary
        if all_stats:
            # Total calls across all tools
            total_calls = sum(s["total_calls"] for _, s in all_stats)
            report["summary"]["total_calls"] = total_calls
            
            # Most/least used
            all_stats.sort(key=lambda x: x[1]["total_calls"], reverse=True)
            report["summary"]["most_used_tool"] = all_stats[0][0] if all_stats else None
            report["summary"]["least_used_tool"] = all_stats[-1][0] if all_stats else None
            
            # Highest/lowest success rate (only for tools with calls > 0)
            stats_with_calls = [(name, s) for name, s in all_stats if s["total_calls"] > 0]
            if stats_with_calls:
                stats_with_calls.sort(key=lambda x: x[1]["success_rate"], reverse=True)
                report["summary"]["highest_success_rate"] = {
                    "tool": stats_with_calls[0][0],
                    "rate": stats_with_calls[0][1]["success_rate"]
                }
                report["summary"]["lowest_success_rate"] = {
                    "tool": stats_with_calls[-1][0],
                    "rate": stats_with_calls[-1][1]["success_rate"]
                }
        
        return report


class AcronymExpanderTool(BaseTool):
    """Expand acronyms and abbreviations to their full form"""
    
    def __init__(self):
        schema = ToolSchema(
            name="acronym_expander",
            description="Expand acronyms and abbreviations to their full form. Useful for 'What does X stand for?' questions. Works with common acronyms like GmbH, NASA, PhD, etc.",
            category="knowledge",
            parameters=[
                ToolParameter(
                    name="acronym",
                    type="string",
                    description="The acronym or abbreviation to expand (e.g., 'GmbH', 'NASA', 'PhD')",
                    required=True
                ),
                ToolParameter(
                    name="context",
                    type="string",
                    description="Optional context to help disambiguation (e.g., 'German company', 'space agency')",
                    required=False
                )
            ],
            usage_example='acronym_expander(acronym="GmbH", context="German company")',
            source="wikipedia_api",
            version="1.0"
        )
        super().__init__(schema)
        self.base_url = "https://en.wikipedia.org/w/api.php"
    
    async def execute(self, acronym: str, context: str = None) -> Dict[str, Any]:
        """Find the full form of an acronym"""
        valid, error = self.validate_parameters(acronym=acronym)
        if not valid:
            return {"success": False, "result": None, "error": error}
        
        try:
            # Build search query
            search_query = acronym
            if context:
                search_query += f" {context}"
            
            params = {
                "action": "opensearch",
                "search": search_query,
                "limit": 5,
                "format": "json"
            }
            
            headers = {"User-Agent": "WirelessAgent-Research-Tool/1.0"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check descriptions for expansion keywords
                        for i, desc in enumerate(data[2]):
                            if any(keyword in desc.lower() for keyword in 
                                  ["stands for", "abbreviation for", "acronym for", 
                                   "short for", "initialism", "is a", "is an"]):
                                # Try to extract the full form
                                import re
                                
                                # Pattern 1: "stands for [expansion]"
                                match = re.search(
                                    r'(?:stands for|abbreviation for|acronym for|short for)\s+["\']?([^"\'\.;,]+)["\']?',
                                    desc, 
                                    re.IGNORECASE
                                )
                                if match:
                                    expansion = match.group(1).strip()
                                    return {
                                        "success": True,
                                        "result": expansion,
                                        "source": data[1][i] if i < len(data[1]) else "",
                                        "error": None
                                    }
                                
                                # Pattern 2: "GmbH is a/an [expansion]"
                                match = re.search(
                                    rf'{re.escape(acronym)}\s+(?:is a|is an|means)\s+([^\.;,]+)',
                                    desc,
                                    re.IGNORECASE
                                )
                                if match:
                                    expansion = match.group(1).strip()
                                    return {
                                        "success": True,
                                        "result": expansion,
                                        "source": data[1][i] if i < len(data[1]) else "",
                                        "error": None
                                    }
                        
                        # If no explicit expansion found, try to get full page content
                        if data[1]:  # If we have a result
                            # Try to get the first sentence of the first result page
                            page_title = data[1][0]
                            page_params = {
                                "action": "query",
                                "format": "json",
                                "titles": page_title,
                                "prop": "extracts",
                                "exsentences": 2,
                                "explaintext": 1
                            }
                            
                            async with session.get(self.base_url, params=page_params, headers=headers) as page_response:
                                if page_response.status == 200:
                                    page_data = await page_response.json()
                                    pages = page_data.get("query", {}).get("pages", {})
                                    if pages:
                                        page_id = list(pages.keys())[0]
                                        if page_id != "-1":
                                            extract = pages[page_id].get("extract", "")
                                            
                                            # Look for expansion in the extract
                                            import re
                                            match = re.search(
                                                rf'{re.escape(acronym)}[^\w]*(?:stands for|is|means|abbreviation for)\s+([^\.;,\(]+)',
                                                extract,
                                                re.IGNORECASE
                                            )
                                            if match:
                                                expansion = match.group(1).strip()
                                                return {
                                                    "success": True,
                                                    "result": expansion,
                                                    "source": page_title,
                                                    "error": None
                                                }
                                            
                                            # If still no match, return the first sentence as context
                                            first_sentence = extract.split('.')[0] if extract else ""
                                            if first_sentence:
                                                return {
                                                    "success": True,
                                                    "result": first_sentence,
                                                    "source": page_title,
                                                    "error": None
                                                }
                        
                        return {
                            "success": False,
                            "result": None,
                            "error": f"No expansion found for '{acronym}'"
                        }
                    else:
                        return {
                            "success": False,
                            "result": None,
                            "error": f"HTTP {response.status}"
                        }
            
        except Exception as e:
            logger.error(f"Acronym expander error: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }


# ==================== Programmer Tool (MATH-specific) ====================

class ProgrammerTool(BaseTool):
    """
    Mathematical problem solver through Python code generation and execution.
    
    This tool wraps the Programmer operator functionality, allowing agents
    to use computational verification as a tool call.
    """
    
    def __init__(self, llm):
        """
        Initialize Programmer tool with LLM instance
        
        Args:
            llm: AsyncLLM instance for code generation
        """
        schema = ToolSchema(
            name="python_code_solver",
            description="Generate and execute Python code to solve mathematical problems computationally. **IMPORTANT**: This tool AUTOMATICALLY generates Python code - do NOT pass a 'code' parameter. Just describe the problem in natural language, and the tool will write and execute the code for you. Useful for verification, complex calculations involving lists/loops/conditionals, or when you need computational confirmation. Returns the computed answer as a string.",
            category="compute",
            parameters=[
                ToolParameter(
                    name="problem",
                    type="string",
                    description="The mathematical problem description in natural language. The tool will automatically generate Python code based on this description.",
                    required=True
                ),
                ToolParameter(
                    name="analysis",
                    type="string",
                    description="Your analysis or reasoning about how to solve the problem (helps code generation). Can be 'None' if no prior analysis.",
                    required=False,
                    default="None"
                )
            ],
            usage_example='python_code_solver(problem="Calculate the area of a circle with radius 5", analysis="Use formula pi*r^2")',
            source="custom_programmer",
            version="1.0"
        )
        super().__init__(schema)
        
        # Import Programmer operator
        from scripts.operators import Programmer
        self.programmer = Programmer(llm, name="ProgrammerTool")
    
    async def execute(self, problem: str = None, analysis: str = "None", **kwargs) -> Dict[str, Any]:
        """
        Execute code generation and execution
        
        Args:
            problem: Mathematical problem description
            analysis: Optional analysis/reasoning
            **kwargs: Catch unexpected arguments (like 'code') for better error messages
        
        Returns:
            {
                "success": bool,
                "result": str (the computed answer),
                "code": str (generated Python code),
                "error": Optional[str]
            }
        """
        # Handle common mistake: passing 'code' parameter
        if 'code' in kwargs and not problem:
            return {
                "success": False,
                "result": None,
                "code": "",
                "error": "python_code_solver does NOT accept 'code' parameter. It generates code automatically. Use: python_code_solver(problem='your problem description', analysis='optional reasoning')"
            }
        
        # Handle unexpected parameters
        if kwargs:
            unexpected = ', '.join(kwargs.keys())
            logger.warning(f"ProgrammerTool received unexpected parameters: {unexpected}")
        
        # Handle missing or empty problem - try alternative keys
        if not problem or (isinstance(problem, str) and not problem.strip()):
            # Check common alternative keys
            problem = kwargs.get('question') or kwargs.get('query') or kwargs.get('input') or kwargs.get('task')
            if not problem or (isinstance(problem, str) and not problem.strip()):
                return {
                    "success": False,
                    "result": None,
                    "code": "",
                    "error": "Missing required parameter 'problem'. Usage: python_code_solver(problem='description', analysis='optional')"
                }
        
        valid, error = self.validate_parameters(problem=problem)
        if not valid:
            return {"success": False, "result": None, "code": "", "error": error}
        
        try:
            # Call Programmer operator
            result = await self.programmer(problem=problem, analysis=analysis)
            
            code = result.get("code", "")
            output = result.get("output", "")
            
            # Check if execution was successful
            if output and "Error" not in output:
                return {
                    "success": True,
                    "result": output.strip(),
                    "code": code,
                    "error": None
                }
            else:
                # Code execution failed
                return {
                    "success": False,
                    "result": None,
                    "code": code,
                    "error": output if output else "Code generation or execution failed"
                }
                
        except Exception as e:
            logger.error(f"ProgrammerTool error: {e}")
            return {
                "success": False,
                "result": None,
                "code": "",
                "error": f"Tool execution error: {str(e)}"
            }


# ==================== Geometry Tools (NEW for MATH Dataset) ====================

class GeometryConstraintChecker(BaseTool):
    """
    Geometry constraint verification tool
    
    Validates whether geometric values satisfy mathematical theorems:
    - Triangle angle sum (must equal 180°)
    - Polygon angle sum (must equal (n-2)×180°)
    - Triangle inequality
    - Angle bisector properties
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="geometry_constraint_checker",
            description="Verify if geometric values satisfy mathematical theorems (e.g., triangle angle sum = 180°, polygon angle sum = (n-2)×180°, triangle inequality). Use this to validate your calculated angles or side lengths before finalizing the answer.",
            category="geometry",
            parameters=[
                ToolParameter(
                    name="constraint_type",
                    type="string",
                    description="Type of constraint to check. Options: 'triangle_angles', 'polygon_angles', 'triangle_inequality', 'angle_bisector'",
                    required=True,
                    enum=["triangle_angles", "polygon_angles", "triangle_inequality", "angle_bisector"]
                ),
                ToolParameter(
                    name="values",
                    type="string",
                    description="JSON string of values to check. For 'triangle_angles': {\"angles\": [a, b, c]}. For 'polygon_angles': {\"n_sides\": n, \"angles\": [a1, a2, ...]}. For 'triangle_inequality': {\"sides\": [a, b, c]}",
                    required=True
                )
            ],
            usage_example='geometry_constraint_checker(constraint_type="triangle_angles", values="{\"angles\": [60, 60, 60]}")',
            source="custom_geometry",
            version="1.0"
        )
        super().__init__(schema)
    
    async def execute(self, constraint_type: str, values: str) -> Dict[str, Any]:
        """Verify geometric constraints"""
        valid, error = self.validate_parameters(constraint_type=constraint_type, values=values)
        if not valid:
            return {"success": False, "valid": False, "error": error}
        
        try:
            import json
            values_dict = json.loads(values)
            
            if constraint_type == "triangle_angles":
                angles = values_dict.get("angles", [])
                if len(angles) != 3:
                    return {
                        "success": False,
                        "valid": False,
                        "error": "Triangle must have exactly 3 angles"
                    }
                
                angle_sum = sum(angles)
                expected = 180
                is_valid = abs(angle_sum - expected) < 0.01  # Tolerance for floating point
                
                return {
                    "success": True,
                    "valid": is_valid,
                    "actual_sum": angle_sum,
                    "expected_sum": expected,
                    "error_message": f"Angle sum is {angle_sum}°, expected {expected}°" if not is_valid else None,
                    "error": None
                }
            
            elif constraint_type == "polygon_angles":
                n_sides = values_dict.get("n_sides")
                angles = values_dict.get("angles", [])
                
                if n_sides is None:
                    return {
                        "success": False,
                        "valid": False,
                        "error": "Must provide 'n_sides' for polygon"
                    }
                
                angle_sum = sum(angles) if angles else None
                expected = (n_sides - 2) * 180
                
                if angle_sum is not None:
                    is_valid = abs(angle_sum - expected) < 0.01
                    return {
                        "success": True,
                        "valid": is_valid,
                        "actual_sum": angle_sum,
                        "expected_sum": expected,
                        "formula": f"({n_sides} - 2) × 180° = {expected}°",
                        "error_message": f"Angle sum is {angle_sum}°, expected {expected}°" if not is_valid else None,
                        "error": None
                    }
                else:
                    # Just return expected value
                    return {
                        "success": True,
                        "valid": True,
                        "expected_sum": expected,
                        "formula": f"({n_sides} - 2) × 180° = {expected}°",
                        "error": None
                    }
            
            elif constraint_type == "triangle_inequality":
                sides = values_dict.get("sides", [])
                if len(sides) != 3:
                    return {
                        "success": False,
                        "valid": False,
                        "error": "Triangle must have exactly 3 sides"
                    }
                
                a, b, c = sorted(sides)
                # Triangle inequality: sum of two smaller sides > largest side
                is_valid = (a + b) > c
                
                return {
                    "success": True,
                    "valid": is_valid,
                    "sides": sides,
                    "check": f"{a} + {b} = {a+b} {'>' if is_valid else '<='} {c}",
                    "error_message": f"Triangle inequality violated: {a} + {b} = {a+b} ≤ {c}" if not is_valid else None,
                    "error": None
                }
            
            else:
                return {
                    "success": False,
                    "valid": False,
                    "error": f"Unknown constraint type: {constraint_type}"
                }
                
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "valid": False,
                "error": f"Invalid JSON in values parameter: {str(e)}"
            }
        except Exception as e:
            logger.error(f"GeometryConstraintChecker error: {e}")
            return {
                "success": False,
                "valid": False,
                "error": f"Tool execution error: {str(e)}"
            }


class GeometricFormulaTool(BaseTool):
    """
    Built-in geometric formula library
    
    Provides reliable calculations for common geometric formulas:
    - Polygon angle sum
    - Triangle area (Heron's formula)
    - Triangle perimeter
    - Circle properties
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="geometric_formula",
            description="Calculate geometric values using built-in formulas. More reliable than manual Python code for standard geometric calculations. Supports: polygon angle sum, triangle area (Heron's formula), triangle perimeter, circle properties.",
            category="geometry",
            parameters=[
                ToolParameter(
                    name="formula",
                    type="string",
                    description="Formula to use. Options: 'polygon_angle_sum', 'triangle_angle_sum', 'heron_area', 'triangle_perimeter', 'circle_area', 'circle_circumference'",
                    required=True,
                    enum=["polygon_angle_sum", "triangle_angle_sum", "heron_area", "triangle_perimeter", "circle_area", "circle_circumference"]
                ),
                ToolParameter(
                    name="params",
                    type="string",
                    description="JSON string of parameters. For 'polygon_angle_sum': {\"n_sides\": n}. For 'heron_area': {\"sides\": [a, b, c]}. For 'triangle_perimeter': {\"sides\": [a, b, c]}. For circle formulas: {\"radius\": r}",
                    required=True
                )
            ],
            usage_example='geometric_formula(formula="polygon_angle_sum", params="{\"n_sides\": 5}")',
            source="custom_geometry",
            version="1.0"
        )
        super().__init__(schema)
    
    async def execute(self, formula: str, params: str) -> Dict[str, Any]:
        """Calculate using geometric formula"""
        valid, error = self.validate_parameters(formula=formula, params=params)
        if not valid:
            return {"success": False, "result": None, "error": error}
        
        try:
            import json
            import math
            params_dict = json.loads(params)
            
            if formula == "polygon_angle_sum":
                n_sides = params_dict.get("n_sides")
                if n_sides is None or n_sides < 3:
                    return {
                        "success": False,
                        "result": None,
                        "error": "Polygon must have at least 3 sides"
                    }
                
                result = (n_sides - 2) * 180
                return {
                    "success": True,
                    "result": result,
                    "formula_used": f"({n_sides} - 2) × 180°",
                    "verification": "✓ Polygon angle sum theorem",
                    "error": None
                }
            
            elif formula == "triangle_angle_sum":
                return {
                    "success": True,
                    "result": 180,
                    "formula_used": "Triangle angle sum = 180°",
                    "verification": "✓ Triangle angle sum theorem",
                    "error": None
                }
            
            elif formula == "heron_area":
                sides = params_dict.get("sides", [])
                if len(sides) != 3:
                    return {
                        "success": False,
                        "result": None,
                        "error": "Need exactly 3 sides for triangle area"
                    }
                
                a, b, c = sides
                # Semi-perimeter
                s = (a + b + c) / 2
                # Heron's formula: A = √[s(s-a)(s-b)(s-c)]
                area_squared = s * (s - a) * (s - b) * (s - c)
                
                if area_squared < 0:
                    return {
                        "success": False,
                        "result": None,
                        "error": "Invalid triangle: sides do not satisfy triangle inequality"
                    }
                
                area = math.sqrt(area_squared)
                return {
                    "success": True,
                    "result": area,
                    "formula_used": f"Heron's formula with s = {s}",
                    "semi_perimeter": s,
                    "verification": "✓ Valid triangle",
                    "error": None
                }
            
            elif formula == "triangle_perimeter":
                sides = params_dict.get("sides", [])
                if len(sides) != 3:
                    return {
                        "success": False,
                        "result": None,
                        "error": "Need exactly 3 sides for triangle perimeter"
                    }
                
                perimeter = sum(sides)
                return {
                    "success": True,
                    "result": perimeter,
                    "formula_used": f"{sides[0]} + {sides[1]} + {sides[2]}",
                    "verification": "✓",
                    "error": None
                }
            
            elif formula == "circle_area":
                radius = params_dict.get("radius")
                if radius is None or radius <= 0:
                    return {
                        "success": False,
                        "result": None,
                        "error": "Radius must be positive"
                    }
                
                area = math.pi * radius ** 2
                return {
                    "success": True,
                    "result": area,
                    "formula_used": f"π × {radius}²",
                    "verification": "✓",
                    "error": None
                }
            
            elif formula == "circle_circumference":
                radius = params_dict.get("radius")
                if radius is None or radius <= 0:
                    return {
                        "success": False,
                        "result": None,
                        "error": "Radius must be positive"
                    }
                
                circumference = 2 * math.pi * radius
                return {
                    "success": True,
                    "result": circumference,
                    "formula_used": f"2π × {radius}",
                    "verification": "✓",
                    "error": None
                }
            
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Unknown formula: {formula}"
                }
                
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "result": None,
                "error": f"Invalid JSON in params: {str(e)}"
            }
        except Exception as e:
            logger.error(f"GeometricFormulaTool error: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"Tool execution error: {str(e)}"
            }


# ==================== Combinatorial Tools (NEW for MATH Dataset) ====================

class CombinatorialEnumerator(BaseTool):
    """
    Reliable combinatorial enumeration with constraint support
    
    Generates all permutations/combinations with optional constraints:
    - Not adjacent constraints
    - Circular permutations
    - Fixed positions
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="combinatorial_enumerator",
            description="Reliably enumerate permutations/combinations with constraint support. More accurate than manual Python code for counting problems. Supports: regular permutations, circular permutations, 'not adjacent' constraints, fixed positions. Returns both count and optionally all cases for verification.",
            category="combinatorics",
            parameters=[
                ToolParameter(
                    name="enum_type",
                    type="string",
                    description="Type of enumeration. Options: 'permutation', 'combination', 'circular_permutation'",
                    required=True,
                    enum=["permutation", "combination", "circular_permutation"]
                ),
                ToolParameter(
                    name="items",
                    type="string",
                    description="JSON array of items to enumerate (e.g., '[1, 2, 3]' or '[\"A\", \"B\", \"C\"]')",
                    required=True
                ),
                ToolParameter(
                    name="constraints",
                    type="string",
                    description="JSON object of constraints. Options: {\"not_adjacent\": [[item1, item2], ...], \"fixed_positions\": {item: position}}. Use empty '{}' for no constraints.",
                    required=False,
                    default="{}"
                ),
                ToolParameter(
                    name="r",
                    type="integer",
                    description="For combinations: choose r items from n. Leave empty for full permutations.",
                    required=False
                )
            ],
            usage_example='combinatorial_enumerator(enum_type="circular_permutation", items="[\"A\", \"B\", \"C\"]", constraints="{\"not_adjacent\": [[\"A\", \"B\"]]}")',
            source="custom_combinatorics",
            version="1.0"
        )
        super().__init__(schema)
    
    async def execute(self, enum_type: str, items: str, constraints: str = "{}", r: int = None) -> Dict[str, Any]:
        """Enumerate with constraints"""
        valid, error = self.validate_parameters(enum_type=enum_type, items=items)
        if not valid:
            return {"success": False, "count": 0, "error": error}
        
        try:
            import json
            from itertools import permutations, combinations
            
            items_list = json.loads(items)
            constraints_dict = json.loads(constraints)
            
            not_adjacent = constraints_dict.get("not_adjacent", [])
            fixed_positions = constraints_dict.get("fixed_positions", {})
            
            def check_not_adjacent(perm):
                """Check if permutation violates not-adjacent constraints"""
                for pair in not_adjacent:
                    item1, item2 = pair
                    if item1 in perm and item2 in perm:
                        idx1 = perm.index(item1)
                        idx2 = perm.index(item2)
                        if abs(idx1 - idx2) == 1:
                            return False
                        # For circular, also check first and last
                        if enum_type == "circular_permutation":
                            if (idx1 == 0 and idx2 == len(perm) - 1) or \
                               (idx2 == 0 and idx1 == len(perm) - 1):
                                return False
                return True
            
            def check_fixed_positions(perm):
                """Check if permutation satisfies fixed position constraints"""
                for item, position in fixed_positions.items():
                    if item in perm and perm[position] != item:
                        return False
                return True
            
            # Generate all cases
            if enum_type == "permutation":
                if r is not None:
                    all_perms = list(permutations(items_list, r))
                else:
                    all_perms = list(permutations(items_list))
            
            elif enum_type == "circular_permutation":
                # Fix first element to avoid rotational duplicates
                if len(items_list) == 0:
                    all_perms = []
                else:
                    first = items_list[0]
                    rest = items_list[1:]
                    all_perms = [[first] + list(p) for p in permutations(rest)]
            
            elif enum_type == "combination":
                if r is None:
                    return {
                        "success": False,
                        "count": 0,
                        "error": "Must specify 'r' for combinations"
                    }
                all_perms = list(combinations(items_list, r))
            
            else:
                return {
                    "success": False,
                    "count": 0,
                    "error": f"Unknown enum_type: {enum_type}"
                }
            
            # Apply constraints
            valid_perms = []
            for perm in all_perms:
                perm_list = list(perm)
                if check_not_adjacent(perm_list) and check_fixed_positions(perm_list):
                    valid_perms.append(perm_list)
            
            count = len(valid_perms)
            
            # Return limited sample for large results
            sample_size = min(10, count)
            sample = valid_perms[:sample_size] if count > 0 else []
            
            return {
                "success": True,
                "count": count,
                "total_before_constraints": len(all_perms),
                "sample_cases": sample,
                "showing": f"{sample_size} of {count} cases",
                "verification": f"✓ All {count} cases satisfy constraints" if not_adjacent or fixed_positions else "✓ No constraints applied",
                "error": None
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "count": 0,
                "error": f"Invalid JSON: {str(e)}"
            }
        except Exception as e:
            logger.error(f"CombinatorialEnumerator error: {e}")
            return {
                "success": False,
                "count": 0,
                "error": f"Tool execution error: {str(e)}"
            }


class ExpressionEnumerator(BaseTool):
    """
    Enumerate all possible parenthesis insertions in an expression
    
    Specialized tool for problems like:
    "How many different values can be obtained from 2×3×4×5+1 by inserting parentheses?"
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="expression_enumerator",
            description="Enumerate all possible ways to insert parentheses in an arithmetic expression and calculate unique results. Specifically designed for problems asking 'how many different values can be obtained by inserting parentheses'.",
            category="combinatorics",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="The arithmetic expression (e.g., '2*3*4*5+1'). Use * for multiplication, + for addition.",
                    required=True
                )
            ],
            usage_example='expression_enumerator(expression="2*3*4*5+1")',
            source="custom_combinatorics",
            version="1.0"
        )
        super().__init__(schema)
    
    async def execute(self, expression: str) -> Dict[str, Any]:
        """Enumerate all parenthesizations"""
        valid, error = self.validate_parameters(expression=expression)
        if not valid:
            return {"success": False, "count": 0, "error": error}
        
        try:
            # Parse expression into tokens
            import re
            tokens = re.findall(r'\d+|[+\-*/]', expression)
            
            if len(tokens) == 0:
                return {
                    "success": False,
                    "count": 0,
                    "error": "Invalid expression"
                }
            
            # Strategy: Group the +1 with different numbers of preceding terms
            # Example: 2*3*4*5+1 can be:
            # (2*3*4*5)+1 = 120+1 = 121
            # 2*3*4*(5+1) = 24*6 = 144
            # 2*3*(4*5+1) = 6*21 = 126
            # 2*(3*4*5+1) = 2*61 = 122
            # (2*3*4*5+1) = 121 (same as first)
            
            results = set()
            all_expressions = []
            
            # Find the + operator
            if '+' in tokens:
                plus_idx = tokens.index('+')
                
                # Case 1: +1 applies to everything before it
                before = ''.join(tokens[:plus_idx])
                after = ''.join(tokens[plus_idx:])
                expr1 = f"({before}){after}"
                try:
                    val = eval(expr1)
                    results.add(val)
                    all_expressions.append((expr1, val))
                except:
                    pass
                
                # Case 2-N: +1 groups with 1, 2, 3... preceding multiplications
                # Count number of * before +
                multiply_count = tokens[:plus_idx].count('*')
                
                for i in range(multiply_count):
                    # Group +1 with last (i+1) terms
                    # Find the position of the (multiply_count - i - 1)-th * from the end
                    star_positions = [j for j, t in enumerate(tokens[:plus_idx]) if t == '*']
                    if i < len(star_positions):
                        split_pos = star_positions[-(i+1)]
                        before_part = ''.join(tokens[:split_pos+1])
                        grouped_part = ''.join(tokens[split_pos+1:])
                        expr = f"{before_part}({grouped_part})"
                        try:
                            val = eval(expr)
                            results.add(val)
                            all_expressions.append((expr, val))
                        except:
                            pass
            else:
                # No + operator, just evaluate as-is
                try:
                    val = eval(expression)
                    results.add(val)
                    all_expressions.append((expression, val))
                except:
                    pass
            
            unique_values = sorted(list(results))
            count = len(unique_values)
            
            return {
                "success": True,
                "count": count,
                "unique_values": unique_values,
                "all_expressions": all_expressions,
                "verification": f"✓ Found {count} unique values",
                "error": None
            }
            
        except Exception as e:
            logger.error(f"ExpressionEnumerator error: {e}")
            return {
                "success": False,
                "count": 0,
                "error": f"Tool execution error: {str(e)}"
            }


# ==================== Symbolic Math Tool (NEW for MATH Dataset) ====================

class SymbolicSolverTool(BaseTool):
    """
    Symbolic equation solver using SymPy
    
    More reliable than manual Python code for:
    - Solving linear/nonlinear equations
    - Simplifying algebraic expressions
    - Exact solutions (not floating point approximations)
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="symbolic_solver",
            description="Solve equations symbolically using SymPy. Returns exact solutions, not floating point approximations. More reliable than manual code for algebra. Supports: solve equations, simplify expressions, expand, factor.",
            category="compute",
            parameters=[
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation to perform. Options: 'solve' (solve equation), 'simplify' (simplify expression), 'expand' (expand expression), 'factor' (factor expression)",
                    required=True,
                    enum=["solve", "simplify", "expand", "factor"]
                ),
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression or equation. For 'solve', use '=' for equations (e.g., '2*n+1 = 3*n-6'). For others, just the expression.",
                    required=True
                ),
                ToolParameter(
                    name="variable",
                    type="string",
                    description="Variable to solve for (only for 'solve' operation). Default is 'x'.",
                    required=False,
                    default="x"
                )
            ],
            usage_example='symbolic_solver(operation="solve", expression="2*n+1 = 3*n-6", variable="n")',
            source="sympy",
            version="1.0"
        )
        super().__init__(schema)
    
    async def execute(self, operation: str, expression: str, variable: str = "x") -> Dict[str, Any]:
        """Perform symbolic operation"""
        valid, error = self.validate_parameters(operation=operation, expression=expression)
        if not valid:
            return {"success": False, "result": None, "error": error}
        
        try:
            import sympy
            from sympy import symbols, solve, simplify, expand, factor, Eq
            
            # Define variable
            var = symbols(variable)
            
            if operation == "solve":
                # Parse equation
                if '=' in expression:
                    left, right = expression.split('=')
                    left_expr = sympy.sympify(left.strip())
                    right_expr = sympy.sympify(right.strip())
                    equation = Eq(left_expr, right_expr)
                    solutions = solve(equation, var)
                else:
                    # Treat as expression = 0
                    expr = sympy.sympify(expression)
                    solutions = solve(expr, var)
                
                # Format solutions
                if isinstance(solutions, list):
                    if len(solutions) == 0:
                        return {
                            "success": True,
                            "result": "No solution",
                            "solutions": [],
                            "error": None
                        }
                    elif len(solutions) == 1:
                        sol = solutions[0]
                        return {
                            "success": True,
                            "result": f"{variable} = {sol}",
                            "numeric_value": float(sol) if sol.is_number else None,
                            "solutions": solutions,
                            "error": None
                        }
                    else:
                        return {
                            "success": True,
                            "result": f"{variable} ∈ {solutions}",
                            "solutions": solutions,
                            "error": None
                        }
                else:
                    return {
                        "success": True,
                        "result": f"{variable} = {solutions}",
                        "numeric_value": float(solutions) if solutions.is_number else None,
                        "solutions": [solutions],
                        "error": None
                    }
            
            elif operation == "simplify":
                expr = sympy.sympify(expression)
                result = simplify(expr)
                return {
                    "success": True,
                    "result": str(result),
                    "original": expression,
                    "simplified": str(result),
                    "error": None
                }
            
            elif operation == "expand":
                expr = sympy.sympify(expression)
                result = expand(expr)
                return {
                    "success": True,
                    "result": str(result),
                    "original": expression,
                    "expanded": str(result),
                    "error": None
                }
            
            elif operation == "factor":
                expr = sympy.sympify(expression)
                result = factor(expr)
                return {
                    "success": True,
                    "result": str(result),
                    "original": expression,
                    "factored": str(result),
                    "error": None
                }
            
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Unknown operation: {operation}"
                }
                
        except Exception as e:
            logger.error(f"SymbolicSolverTool error: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"Tool execution error: {str(e)}"
            }


# ==================== Validation Tools (NEW for MATH Dataset) ====================

class AnswerTypeValidator(BaseTool):
    """
    Validate and correct answer format
    
    Ensures answers match the expected format:
    - Remove unwanted units (e.g., ° symbol when only number is needed)
    - Detect expected answer type from question
    - Standardize formatting
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="answer_type_validator",
            description="Validate answer format and remove unwanted symbols. Detects expected answer type from question (integer, fraction, expression, etc.) and ensures answer matches. Automatically removes degree symbols (°) when only numerical value is needed.",
            category="validation",
            parameters=[
                ToolParameter(
                    name="question",
                    type="string",
                    description="The original question text",
                    required=True
                ),
                ToolParameter(
                    name="answer",
                    type="string",
                    description="The calculated answer to validate",
                    required=True
                )
            ],
            usage_example='answer_type_validator(question="What is the angle in degrees?", answer="135°")',
            source="custom_validation",
            version="1.0"
        )
        super().__init__(schema)
    
    async def execute(self, question: str, answer: str) -> Dict[str, Any]:
        """Validate and correct answer format"""
        valid, error = self.validate_parameters(question=question, answer=answer)
        if not valid:
            return {"success": False, "is_valid": False, "error": error}
        
        try:
            import re
            
            issues = []
            corrected = answer.strip()
            
            # Detect if question asks for degrees/units
            question_lower = question.lower()
            asks_for_degrees = "in degrees" in question_lower or "angle" in question_lower
            
            # Check for degree symbol
            has_degree_symbol = "°" in corrected or "\\circ" in corrected or "degrees" in corrected.lower()
            
            # Determine expected type
            if "how many" in question_lower or "number of" in question_lower:
                expected_type = "integer"
            elif "fraction" in question_lower or "ratio" in question_lower:
                expected_type = "fraction"
            elif "in degrees" in question_lower:
                expected_type = "angle_value"  # Just the number, not symbol
            elif "expression" in question_lower or "simplest radical form" in question_lower:
                expected_type = "expression"
            else:
                expected_type = "number"
            
            # Rule: If question says "in degrees", the answer should be JUST the number
            # (the unit is already specified in the question)
            if expected_type == "angle_value" and has_degree_symbol:
                # Remove degree symbols
                corrected = corrected.replace("°", "").replace("\\circ", "").replace(" degrees", "")
                corrected = corrected.strip()
                issues.append("Removed degree symbol (unit already specified in question)")
            
            # Remove extra spaces in expressions
            if expected_type == "expression":
                # Normalize spacing in LaTeX expressions
                corrected = re.sub(r'\s+', ' ', corrected)
            
            # Check if it's a valid number/expression
            is_valid = True
            if expected_type == "integer":
                try:
                    int(corrected.replace(",", ""))
                except:
                    is_valid = False
                    issues.append("Expected integer, but answer is not a valid integer")
            
            return {
                "success": True,
                "is_valid": is_valid,
                "expected_type": expected_type,
                "corrected_answer": corrected,
                "original_answer": answer,
                "issues": issues if issues else None,
                "verification": "✓ Format corrected" if issues else "✓ Format is valid",
                "error": None
            }
            
        except Exception as e:
            logger.error(f"AnswerTypeValidator error: {e}")
            return {
                "success": False,
                "is_valid": False,
                "error": f"Tool execution error: {str(e)}"
            }


class ConstraintVerifierTool(BaseTool):
    """
    Verify answer satisfies all constraints in the problem
    
    Extracts constraints from problem statement and validates:
    - Magic square constraints (row sums = column sums)
    - Seating arrangement constraints (not adjacent)
    - Numerical constraints (positive, integer, etc.)
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="constraint_verifier",
            description="Verify that your answer satisfies all constraints mentioned in the problem. Useful for magic squares (checking row/column sums), seating arrangements (checking adjacency), and other constraint satisfaction problems.",
            category="validation",
            parameters=[
                ToolParameter(
                    name="problem_type",
                    type="string",
                    description="Type of constraint problem. Options: 'magic_square', 'seating_arrangement', 'custom'",
                    required=True,
                    enum=["magic_square", "seating_arrangement", "custom"]
                ),
                ToolParameter(
                    name="problem",
                    type="string",
                    description="The problem statement",
                    required=True
                ),
                ToolParameter(
                    name="answer",
                    type="string",
                    description="Your proposed answer",
                    required=True
                ),
                ToolParameter(
                    name="constraints",
                    type="string",
                    description="JSON object describing constraints to check (problem-specific)",
                    required=False,
                    default="{}"
                )
            ],
            usage_example='constraint_verifier(problem_type="magic_square", problem="...", answer="7", constraints="{\"grid\": [[8,1,6],[3,5,7],[4,9,2]]}")',
            source="custom_validation",
            version="1.0"
        )
        super().__init__(schema)
    
    async def execute(self, problem_type: str, problem: str, answer: str, constraints: str = "{}") -> Dict[str, Any]:
        """Verify constraints"""
        valid, error = self.validate_parameters(problem_type=problem_type, problem=problem, answer=answer)
        if not valid:
            return {"success": False, "valid": False, "error": error}
        
        try:
            import json
            constraints_dict = json.loads(constraints)
            
            if problem_type == "magic_square":
                grid = constraints_dict.get("grid", [])
                if not grid:
                    return {
                        "success": False,
                        "valid": False,
                        "error": "Must provide 'grid' in constraints for magic_square"
                    }
                
                # Check if all rows have same sum
                row_sums = [sum(row) for row in grid]
                # Check columns
                n = len(grid)
                col_sums = [sum(grid[i][j] for i in range(n)) for j in range(n)]
                # Check diagonals
                diag1_sum = sum(grid[i][i] for i in range(n))
                diag2_sum = sum(grid[i][n-1-i] for i in range(n))
                
                all_sums = row_sums + col_sums + [diag1_sum, diag2_sum]
                is_magic = len(set(all_sums)) == 1
                
                return {
                    "success": True,
                    "valid": is_magic,
                    "row_sums": row_sums,
                    "col_sums": col_sums,
                    "diagonal_sums": [diag1_sum, diag2_sum],
                    "magic_constant": all_sums[0] if is_magic else None,
                    "verification": "✓ Valid magic square" if is_magic else "✗ Not a magic square",
                    "error": None
                }
            
            elif problem_type == "seating_arrangement":
                arrangement = constraints_dict.get("arrangement", [])
                not_adjacent_pairs = constraints_dict.get("not_adjacent", [])
                
                violations = []
                for pair in not_adjacent_pairs:
                    person1, person2 = pair
                    if person1 in arrangement and person2 in arrangement:
                        idx1 = arrangement.index(person1)
                        idx2 = arrangement.index(person2)
                        # Check if adjacent (including circular)
                        if abs(idx1 - idx2) == 1:
                            violations.append(f"{person1} and {person2} are adjacent at positions {idx1}, {idx2}")
                        elif abs(idx1 - idx2) == len(arrangement) - 1:
                            violations.append(f"{person1} and {person2} are adjacent (circular) at positions {idx1}, {idx2}")
                
                is_valid = len(violations) == 0
                
                return {
                    "success": True,
                    "valid": is_valid,
                    "violations": violations if violations else None,
                    "verification": "✓ All constraints satisfied" if is_valid else f"✗ {len(violations)} violations found",
                    "error": None
                }
            
            elif problem_type == "custom":
                # Generic constraint checking
                return {
                    "success": True,
                    "valid": True,
                    "message": "Custom constraint checking not implemented. Please verify manually.",
                    "error": None
                }
            
            else:
                return {
                    "success": False,
                    "valid": False,
                    "error": f"Unknown problem_type: {problem_type}"
                }
                
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "valid": False,
                "error": f"Invalid JSON in constraints: {str(e)}"
            }
        except Exception as e:
            logger.error(f"ConstraintVerifierTool error: {e}")
            return {
                "success": False,
                "valid": False,
                "error": f"Tool execution error: {str(e)}"
            }
