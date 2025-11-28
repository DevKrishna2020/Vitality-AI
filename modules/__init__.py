"""
modules package — lazy export point for Vitality AI components.

This file exports a small set of top-level helpers while avoiding eager imports
of submodules that may load heavy dependencies at import time.

Usage stays the same:
    from modules import set_page_config, load_model, get_health_tools, create_health_agent, extract_pdf_data
"""

from typing import Dict
import importlib

# Map exported name -> module path that provides it
_EXPORT_MAP: Dict[str, str] = {
    # UI helpers
    "set_page_config": "modules.ui_components",
    "inject_css": "modules.ui_components",
    "init_session": "modules.ui_components",
    # LLM factory
    "load_model": "modules.llm_factory",
    # Health tools
    "get_health_tools": "modules.bio_tools",
    # Agent
    "create_health_agent": "modules.agent_engine",
    # File handling
    "extract_pdf_data": "modules.file_handler",
}

__all__ = list(_EXPORT_MAP.keys())

def __getattr__(name: str):
    """
    Lazy-load the attribute from the mapped submodule on first access.
    Caches the attribute in this module's globals for subsequent access.
    """
    if name in _EXPORT_MAP:
        mod_name = _EXPORT_MAP[name]
        mod = importlib.import_module(mod_name)
        try:
            attr = getattr(mod, name)
        except AttributeError:
            # Fallback: some modules export a differently-named symbol (rare).
            # Try a few common alternative names
            alt_map = {
                "inject_css": "inject_css",
                "init_session": "init_session",
            }
            raise

        # Cache in module globals for faster future access
        globals()[name] = attr
        return attr

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    # Expose lazy-exported names for tab-completion
    return sorted(list(globals().keys()) + __all__)