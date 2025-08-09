import logging
from typing import Any, Dict

def setup_logger() -> logging.Logger:
    """Setup and configure logger"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def format_response(status: str, data: Any = None, message: str = None) -> Dict:
    """
    Format API response in a consistent structure
    
    Args:
        status: Response status ('success' or 'error')
        data: Response data (optional)
        message: Response message (optional)
        
    Returns:
        Dictionary with formatted response
    """
    response = {'status': status}
    if data is not None:
        response['data'] = data
    if message is not None:
        response['message'] = message
    return response
