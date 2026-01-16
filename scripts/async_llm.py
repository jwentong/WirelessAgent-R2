# -*- coding: utf-8 -*-
# @Date    : 1/13/2026
# @Author  : Jingwen
# @Desc    : 

from openai import AsyncOpenAI
from scripts.formatter import BaseFormatter, FormatError

import yaml
from pathlib import Path
from typing import Dict, Optional, Any
import asyncio
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    after_log
)
import logging
from openai import (
    RateLimitError, 
    APIError,
    InternalServerError,
    BadRequestError,
    AuthenticationError
)

logger = logging.getLogger(__name__)


def is_retryable_error(exception):
    """
    Determine if an error should be retried
    
    Retryable errors:
    - RateLimitError (429): Rate limit exceeded
    - InternalServerError (500): Server-side issues
    - APIError: Generic API errors (except BadRequest)
    
    Non-retryable errors:
    - BadRequestError (400): Content inspection failed, invalid input, etc.
    - AuthenticationError (401): Invalid API key
    """
    # Don't retry BadRequest errors (content inspection, invalid params, etc.)
    if isinstance(exception, BadRequestError):
        logger.warning(f"Non-retryable BadRequest error: {exception}")
        return False
    
    # Don't retry authentication errors
    if isinstance(exception, AuthenticationError):
        logger.error(f"Authentication failed: {exception}")
        return False
    
    # Retry rate limits and server errors
    if isinstance(exception, (RateLimitError, InternalServerError)):
        return True
    
    # For generic APIError, check if it's server-side (5xx)
    if isinstance(exception, APIError):
        # Try to extract status code if available
        if hasattr(exception, 'status_code'):
            return exception.status_code >= 500
        return True  # Default to retry for unknown APIErrors
    
    return False

class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key", None)
        # Use provided base_url, only fall back if explicitly None or missing
        self.base_url = config.get("base_url") or "https://oneapi.deepwisdom.ai/v1"
        self.top_p = config.get("top_p", 1)
        self.max_tokens = config.get("max_tokens", 256)  # Limit output for faster response
        logger.info(f"LLMConfig initialized: model={self.model}, base_url={self.base_url}, max_tokens={self.max_tokens}")

class LLMsConfig:
    """Configuration manager for multiple LLM configurations"""
    
    _instance = None  # For singleton pattern if needed
    _default_config = None
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with an optional configuration dictionary"""
        self.configs = config_dict or {}
    
    @classmethod
    def default(cls):
        """Get or create a default configuration from YAML file"""
        if cls._default_config is None:
            # Look for the config file in common locations
            config_paths = [
                Path("config/config2.yaml"),
                Path("config2.yaml"),
                Path("./config/config2.yaml"),
                Path("/app/config/config2.yaml"),  # Docker path
            ]
            
            config_file = None
            for path in config_paths:
                logger.debug(f"Checking config path: {path} (exists: {path.exists()})")
                if path.exists():
                    config_file = path
                    break
            
            if config_file is None:
                logger.error(f"No default configuration file found. Tried: {config_paths}")
                raise FileNotFoundError("No default configuration file found in the expected locations")
            
            # Load the YAML file
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Your YAML has a 'models' top-level key that contains the model configs
            if 'models' in config_data:
                config_data = config_data['models']
                
            cls._default_config = cls(config_data)
        
        return cls._default_config
    
    def get(self, llm_name: str) -> LLMConfig:
        """Get the configuration for a specific LLM by name"""
        if llm_name not in self.configs:
            logger.error(f"Configuration for {llm_name} not found. Available: {list(self.configs.keys())}")
            raise ValueError(f"Configuration for {llm_name} not found")
        
        config = self.configs[llm_name]
        
        # Create a config dictionary with the expected keys for LLMConfig
        llm_config = {
            "model": llm_name,  # Use the key as the model name
            "temperature": config.get("temperature", 1),
            "key": config.get("api_key"),  # Map api_key to key
            "base_url": config.get("base_url"),
            "top_p": config.get("top_p", 1)  # Add top_p parameter
        }
        
        logger.info(f"Loaded LLM config for {llm_name}: base_url={llm_config['base_url']}")
        
        # Create and return an LLMConfig instance with the specified configuration
        return LLMConfig(llm_config)
    
    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add or update a configuration"""
        self.configs[name] = config
    
    def get_all_names(self) -> list:
        """Get names of all available LLM configurations"""
        return list(self.configs.keys())
    
class ModelPricing:
    """Pricing information for different models in USD (1 USD = 6 RMB) per 1K tokens"""
    PRICES = {
        # GPT-4o models
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-4-5-sonnet": {"input": 0.00255, "output": 0.01275},
        "qwen-turbo-latest": {"input": 0.00005, "output": 0.0006},
        "o3":{"input":0.003, "output":0.015},
        "o3-mini": {"input": 0.0011, "output": 0.0025},
    }
    
    @classmethod
    def get_price(cls, model_name, token_type):
        """Get the price per 1K tokens for a specific model and token type (input/output)"""
        # Try to find exact match first
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]
        
        # Try to find a partial match (e.g., if model name contains version numbers)
        for key in cls.PRICES:
            if key in model_name:
                return cls.PRICES[key][token_type]
        
        # Return default pricing if no match found
        return 0

class TokenUsageTracker:
    """Tracks token usage and calculates costs"""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []
    
    def add_usage(self, model, input_tokens, output_tokens):
        """Add token usage for a specific API call"""
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        total_cost = input_cost + output_cost
        
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "prices": {
                "input_price": ModelPricing.get_price(model, "input"),
                "output_price": ModelPricing.get_price(model, "output")
            }
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        self.usage_history.append(usage_record)
        
        return usage_record
    
    def get_summary(self):
        """Get a summary of token usage and costs"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }

class AsyncLLM:
    def __init__(self, config, system_msg:str = None):
        """
        Initialize the AsyncLLM with a configuration
        
        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
        """
        # Handle the case where config is a string (LLM name)
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)
        
        # At this point, config should be an LLMConfig instance
        self.config = config
        self.aclient = AsyncOpenAI(api_key=self.config.key, base_url=self.config.base_url)
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()
        
        # Rate limiting: track last call time to add delays
        self._last_call_time = 0
        self._min_interval = 0.1  # Minimum 0.1 seconds between calls (minimal delay)
        self._call_count = 0  # Track total API calls for debugging
        self._rate_limit_backoff = 1.0  # Dynamic backoff multiplier
        
    @retry(
        stop=stop_after_attempt(1),  # NO retries - fail fast to avoid timeout
        wait=wait_exponential(multiplier=1, min=1, max=5),  # Shorter waits: 1, 2, 4, 5 seconds
        retry=is_retryable_error,  # Use custom retry logic (only retry recoverable errors)
        before_sleep=before_sleep_log(logger, logging.WARNING),  # Log before sleeping
        after=after_log(logger, logging.INFO)  # Log after attempt
    )
    async def __call__(self, prompt):
        # Increment call counter
        self._call_count += 1
        
        # Rate limiting: ensure minimum interval between calls with dynamic backoff
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        effective_interval = self._min_interval * self._rate_limit_backoff
        
        if time_since_last_call < effective_interval:
            sleep_time = effective_interval - time_since_last_call
            logger.info(f"⏱️ Rate limiting: sleeping {sleep_time:.2f}s (call #{self._call_count}, backoff: {self._rate_limit_backoff:.1f}x)")
            await asyncio.sleep(sleep_time)
        
        self._last_call_time = time.time()
        
        logger.debug(f"API Call #{self._call_count} to {self.config.model}")
        
        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })

        message.append({"role": "user", "content": prompt})

        try:
            response = await self.aclient.chat.completions.create(
                model=self.config.model,
                messages=message,
                temperature=self.config.temperature,
                top_p = self.config.top_p,
                max_tokens=self.config.max_tokens,  # Limit output tokens for faster response
            )
            
            # Successful call - gradually reduce backoff
            if self._rate_limit_backoff > 1.0:
                self._rate_limit_backoff = max(1.0, self._rate_limit_backoff * 0.9)
                logger.debug(f"Reducing rate limit backoff to {self._rate_limit_backoff:.2f}x")
                
        except BadRequestError as e:
            # Don't retry BadRequest errors (content inspection, invalid input, etc.)
            logger.error(f"BadRequest error (400): {e}. This is not retryable.")
            logger.warning("Possible causes: content inspection failed, invalid parameters, or inappropriate content.")
            raise  # Re-raise immediately, tenacity won't retry due to is_retryable_error()
        except RateLimitError as e:
            # Increase backoff for future calls
            self._rate_limit_backoff = min(5.0, self._rate_limit_backoff * 1.5)
            logger.error(f"Rate limit error (429): {e}. Increasing backoff to {self._rate_limit_backoff:.2f}x. Will retry with exponential backoff...")
            raise  # Re-raise to trigger tenacity retry
        except InternalServerError as e:
            logger.error(f"Server error (500): {e}. Will retry...")
            raise
        except APIError as e:
            logger.error(f"API error: {e}. Will retry if it's a server-side issue...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM call: {e}")
            raise

        # Extract token usage from response
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # Track token usage and calculate cost
        usage_record = self.usage_tracker.add_usage(
            self.config.model,
            input_tokens,
            output_tokens
        )
        
        ret = response.choices[0].message.content
        print(ret)
        
        # You can optionally print token usage information
        print(f"Token usage: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total")
        print(f"Cost: ${usage_record['total_cost']:.6f} (${usage_record['input_cost']:.6f} for input, ${usage_record['output_cost']:.6f} for output)")
        
        return ret
    
    async def call_with_format(self, prompt: str, formatter: BaseFormatter):
        """
        Call the LLM with a prompt and format the response using the provided formatter
        
        Args:
            prompt: The prompt to send to the LLM
            formatter: An instance of a BaseFormatter to validate and parse the response
            
        Returns:
            The formatted response data
            
        Raises:
            FormatError: If the response doesn't match the expected format
        """
        # Prepare the prompt with formatting instructions
        formatted_prompt = formatter.prepare_prompt(prompt)
        # Call the LLM
        response = await self.__call__(formatted_prompt)
        
        # Validate and parse the response
        is_valid, parsed_data = formatter.validate_response(response)
        
        if not is_valid:
            error_message = formatter.format_error_message()
            raise FormatError(f"{error_message}. Raw response: {response}")
        
        return parsed_data
    
    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        return self.usage_tracker.get_summary()    
    

def create_llm_instance(llm_config):
    """
    Create an AsyncLLM instance using the provided configuration
    
    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config
    
    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    # Case 1: llm_config is already an LLMConfig instance
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config)
    
    # Case 2: llm_config is a string (LLM name)
    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config)  # AsyncLLM constructor handles lookup
    
    # Case 3: llm_config is a dictionary
    elif isinstance(llm_config, dict):
        # Create an LLMConfig instance from the dictionary
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config)
    
    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")