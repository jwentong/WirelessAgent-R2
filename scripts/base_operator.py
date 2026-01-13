# -*- coding: utf-8 -*-
# @Date    : 1/13/2026
# @Author  : Jingwen
# @Desc    : Base classes for standardized operator implementation

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import time
from datetime import datetime

from scripts.logs import logger


@dataclass
class OperatorResult:
    """
    Standardized return format for all operators.
    
    This ensures consistent output structure across different operators,
    making it easier to compose workflows and collect metrics.
    
    Attributes:
        output: The primary output of the operator (can be any type)
        cost: LLM cost incurred during execution (in USD)
        metadata: Additional execution information (tool stats, intermediate steps, etc.)
        success: Whether the operation completed successfully
        error: Error message if success=False
        execution_time: Time taken to execute (in seconds)
    """
    output: Any
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "output": self.output,
            "cost": self.cost,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
            "execution_time": self.execution_time
        }


class OperatorMetrics:
    """
    Automatic metrics collection for operators.
    
    Tracks execution statistics to help identify bottlenecks and optimize workflows.
    """
    
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_cost = 0.0
        self.total_time = 0.0
        self.call_history: List[Dict[str, Any]] = []
        
    def record(self, duration: float, cost: float, success: bool, error: Optional[str] = None):
        """Record a single execution"""
        self.total_calls += 1
        self.total_cost += cost
        self.total_time += duration
        
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        # Keep last 100 calls for analysis
        self.call_history.append({
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "cost": cost,
            "success": success,
            "error": error
        })
        if len(self.call_history) > 100:
            self.call_history.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics"""
        success_rate = self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0.0
        avg_cost = self.total_cost / self.total_calls if self.total_calls > 0 else 0.0
        
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": success_rate,
            "total_cost": self.total_cost,
            "total_time": self.total_time,
            "avg_time": avg_time,
            "avg_cost": avg_cost
        }
    
    def reset(self):
        """Reset all metrics"""
        self.__init__()


class BaseOperator(ABC):
    """
    Abstract base class for all operators in WirelessAgent.
    
    This class provides:
    1. Unified interface for operator execution
    2. Automatic metrics collection
    3. Metadata and schema introspection
    4. Consistent error handling
    
    Subclasses should:
    - Implement the _execute() method with core logic
    - Define _get_input_schema() and _get_output_schema() for introspection
    - Use self.llm for LLM calls if needed
    
    Example:
        class MyOperator(BaseOperator):
            async def _execute(self, input_text: str, **kwargs) -> Any:
                result = await self.llm(f"Process: {input_text}")
                return result
            
            def _get_input_schema(self) -> Dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "input_text": {"type": "string"}
                    },
                    "required": ["input_text"]
                }
    """
    
    def __init__(self, llm=None, enable_metrics: bool = True):
        """
        Initialize the operator.
        
        Args:
            llm: Language model instance (optional, not all operators need LLM)
            enable_metrics: Whether to automatically collect metrics
        """
        self.llm = llm
        self.enable_metrics = enable_metrics
        self.metrics = OperatorMetrics() if enable_metrics else None
        self._name = self.__class__.__name__
    
    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """
        Core execution logic to be implemented by subclasses.
        
        This method should contain the actual operator logic.
        Exceptions will be caught and handled by __call__.
        
        Args:
            **kwargs: Operator-specific parameters
            
        Returns:
            The operator's output (any type)
        """
        pass
    
    async def __call__(self, **kwargs) -> Union[OperatorResult, Any]:
        """
        Execute the operator with automatic tracking.
        
        This wrapper:
        - Tracks execution time and cost
        - Handles errors gracefully
        - Records metrics automatically
        - Returns OperatorResult or raw output (for backward compatibility)
        
        Args:
            **kwargs: Operator-specific parameters
            return_result_object: If True, return OperatorResult; if False, return raw output (default: False for backward compatibility)
            
        Returns:
            OperatorResult object or raw output depending on return_result_object flag
        """
        start_time = time.time()
        return_result_object = kwargs.pop('return_result_object', False)
        
        try:
            # Execute the operator
            output = await self._execute(**kwargs)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            cost = self._extract_cost(output)
            
            # Record metrics if enabled
            if self.metrics:
                self.metrics.record(
                    duration=execution_time,
                    cost=cost,
                    success=True
                )
            
            # Build result
            result = OperatorResult(
                output=output,
                cost=cost,
                metadata=self._extract_metadata(output),
                success=True,
                execution_time=execution_time
            )
            
            # Return based on flag
            if return_result_object:
                return result
            else:
                # Backward compatibility: return raw output
                return output
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{self._name} execution failed: {str(e)}"
            
            # Classify error type for appropriate logging
            error_type = type(e).__name__
            error_str = str(e)
            
            # Content inspection errors (expected, system continues)
            is_content_error = (
                "data_inspection_failed" in error_str or
                "inappropriate content" in error_str.lower()
            )
            
            # System/infrastructure errors (need attention)
            is_system_error = (
                "RateLimitError" in error_type or
                "Error code: 429" in error_str or
                "limit_requests" in error_str or
                "ServiceUnavailableError" in error_type or
                "InternalServerError" in error_type
            )
            
            # Log at appropriate level
            if is_content_error:
                logger.warning(error_msg)  # Content issues - expected behavior
            elif is_system_error:
                logger.error(error_msg)  # System issues - need attention
            elif "BadRequestError" in error_type:
                logger.warning(error_msg)  # Bad request - likely content issue
            else:
                logger.error(error_msg)  # Unknown errors - log as error for safety
            
            # Record failure metrics
            if self.metrics:
                self.metrics.record(
                    duration=execution_time,
                    cost=0.0,
                    success=False,
                    error=str(e)
                )
            
            # Build error result
            result = OperatorResult(
                output=None,
                cost=0.0,
                metadata={"error_type": type(e).__name__},
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
            
            if return_result_object:
                return result
            else:
                # Backward compatibility: re-raise exception
                raise
    
    def _extract_cost(self, output: Any) -> float:
        """
        Extract LLM cost from output.
        
        Override this method if your operator has a custom way of tracking cost.
        
        Args:
            output: The operator's output
            
        Returns:
            Cost in USD
        """
        # Try common patterns
        if isinstance(output, dict):
            if 'cost' in output:
                return float(output['cost'])
            if 'total_cost' in output:
                return float(output['total_cost'])
        
        # Try to get from LLM instance
        if self.llm and hasattr(self.llm, 'get_usage_summary'):
            summary = self.llm.get_usage_summary()
            if 'total_cost' in summary:
                return summary['total_cost']
        
        return 0.0
    
    def _extract_metadata(self, output: Any) -> Dict[str, Any]:
        """
        Extract metadata from output.
        
        Override this method to extract operator-specific metadata.
        
        Args:
            output: The operator's output
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        if isinstance(output, dict):
            # Common metadata fields
            for key in ['steps', 'tool_stats', 'intermediate_results', 'flags']:
                if key in output:
                    metadata[key] = output[key]
        
        return metadata
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get operator metadata for introspection.
        
        This is useful for:
        - MCTS optimization (LLM needs to know what operators do)
        - Automatic documentation
        - Tool selection
        
        Returns:
            Metadata including name, description, schemas, and statistics
        """
        return {
            "name": self._name,
            "description": self.__doc__ or "No description available",
            "input_schema": self._get_input_schema(),
            "output_schema": self._get_output_schema(),
            "statistics": self.metrics.get_stats() if self.metrics else None
        }
    
    def _get_input_schema(self) -> Dict[str, Any]:
        """
        Define input parameter schema (JSON Schema format).
        
        Override this method to provide schema for validation and documentation.
        
        Returns:
            JSON Schema dictionary
        """
        return {
            "type": "object",
            "properties": {},
            "description": "No input schema defined"
        }
    
    def _get_output_schema(self) -> Dict[str, Any]:
        """
        Define output schema (JSON Schema format).
        
        Override this method to describe what the operator returns.
        
        Returns:
            JSON Schema dictionary
        """
        return {
            "type": "object",
            "properties": {},
            "description": "No output schema defined"
        }
    
    def get_statistics(self) -> Optional[Dict[str, Any]]:
        """Get execution statistics"""
        return self.metrics.get_stats() if self.metrics else None
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        if self.metrics:
            self.metrics.reset()


class LegacyOperatorWrapper:
    """
    Wrapper to make old-style operators compatible with new BaseOperator interface.
    
    This allows gradual migration without breaking existing workflows.
    
    Example:
        # Old operator
        class OldOperator:
            def __init__(self, llm):
                self.llm = llm
            async def __call__(self, input: str):
                return await self.llm(input)
        
        # Make it compatible
        old_op = OldOperator(llm)
        wrapped_op = LegacyOperatorWrapper(old_op)
        result = await wrapped_op(input="test", return_result_object=True)
        # result is now an OperatorResult
    """
    
    def __init__(self, legacy_operator):
        self.legacy_operator = legacy_operator
        self.metrics = OperatorMetrics()
        self._name = legacy_operator.__class__.__name__
    
    async def __call__(self, **kwargs):
        return_result_object = kwargs.pop('return_result_object', False)
        start_time = time.time()
        
        try:
            output = await self.legacy_operator(**kwargs)
            execution_time = time.time() - start_time
            
            self.metrics.record(execution_time, 0.0, True)
            
            if return_result_object:
                return OperatorResult(
                    output=output,
                    success=True,
                    execution_time=execution_time
                )
            return output
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.record(execution_time, 0.0, False, str(e))
            
            if return_result_object:
                return OperatorResult(
                    output=None,
                    success=False,
                    error=str(e),
                    execution_time=execution_time
                )
            raise
