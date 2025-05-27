import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from models import AppConfig, LLMSiteConfig, GlobalSettings
from utils import setup_logging

# Initialize logger
logger = setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))

# Global config instance
_config: Optional[AppConfig] = None

def load_config(config_path: Union[str, Path] = "config.yaml") -> AppConfig:
    """
    Load configuration from YAML file and validate using Pydantic models.
    
    Args:
        config_path (Union[str, Path]): Path to the configuration file
        
    Returns:
        AppConfig: Validated configuration object
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the configuration is invalid
    """
    global _config
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        error_msg = f"Configuration file not found: {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        # Validate configuration using Pydantic model
        _config = AppConfig(**config_data)
        logger.info("Configuration loaded and validated successfully")
        
        # Log some basic configuration info
        logger.debug(f"Global timeout: {_config.global_settings.timeout}s")
        logger.debug(f"Configured LLM sites: {[site.id for site in _config.llm_sites]}")
        
        return _config
    
    except yaml.YAMLError as e:
        error_msg = f"Error parsing YAML configuration: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    except Exception as e:
        error_msg = f"Error loading configuration: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_config() -> AppConfig:
    """
    Get the current configuration. Loads the default configuration if not already loaded.
    
    Returns:
        AppConfig: The current configuration
    """
    global _config
    
    if _config is None:
        _config = load_config()
    
    return _config

def get_llm_site_config(site_id: str) -> LLMSiteConfig:
    """
    Get configuration for a specific LLM site.
    
    Args:
        site_id (str): ID of the LLM site to get configuration for
        
    Returns:
        LLMSiteConfig: Configuration for the specified LLM site
        
    Raises:
        KeyError: If the site doesn't exist in the configuration
    """
    config = get_config()
    
    for site in config.llm_sites:
        if site.id == site_id:
            if not site.enabled:
                logger.warning(f"LLM site '{site_id}' is disabled in configuration")
            return site
    
    error_msg = f"LLM site '{site_id}' not found in configuration"
    logger.error(error_msg)
    raise KeyError(error_msg)

def get_enabled_llm_sites() -> List[LLMSiteConfig]:
    """
    Get a list of all enabled LLM sites.
    
    Returns:
        List[LLMSiteConfig]: List of enabled LLM site configurations
    """
    config = get_config()
    return [site for site in config.llm_sites if site.enabled]

def get_global_settings() -> GlobalSettings:
    """
    Get global settings from configuration.
    
    Returns:
        GlobalSettings: Global settings object
    """
    config = get_config()
    return config.global_settings

def reload_config(config_path: Union[str, Path] = "config.yaml") -> AppConfig:
    """
    Force reload of configuration from file.
    
    Args:
        config_path (Union[str, Path]): Path to the configuration file
        
    Returns:
        AppConfig: Reloaded configuration object
    """
    global _config
    _config = None
    return load_config(config_path)

# Example usage
if __name__ == "__main__":
    try:
        # Load configuration
        config = get_config()
        print(f"Loaded configuration with {len(config.llm_sites)} LLM sites")
        
        # Get enabled sites
        enabled_sites = get_enabled_llm_sites()
        print(f"Found {len(enabled_sites)} enabled LLM sites")
        
        # Display site information
        for site in enabled_sites:
            print(f"Site '{site.id}': {site.name} ({site.url})")
        
    except Exception as e:
        print(f"Error: {e}")