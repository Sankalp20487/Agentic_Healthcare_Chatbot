"""
Rate limiter implementation using token bucket algorithm.
Prevents API quota exhaustion by limiting request frequency.
"""

import time
from typing import Optional


class TokenBucket:
    """
    Token bucket rate limiter for API calls.
    
    Implements a token bucket algorithm with automatic token refill.
    Used to prevent excessive API calls that could lead to quota issues.
    
    Attributes:
        capacity: Maximum number of tokens the bucket can hold
        tokens: Current number of available tokens
        fill_rate: Tokens added per second
        last_time: Timestamp of last token consumption
    """
    
    def __init__(self, tokens: int, fill_rate: float):
        """
        Initialize token bucket rate limiter.
        
        Args:
            tokens: Initial number of tokens (also the capacity)
            fill_rate: Rate at which tokens are added per second
        """
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.last_time = time.time()
        
    def consume(self, tokens: int = 1) -> float:
        """
        Attempt to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            Wait time needed (in seconds) if insufficient tokens, 0 otherwise
        """
        # Refill tokens based on elapsed time
        now = time.time()
        time_passed = now - self.last_time
        self.tokens = min(self.capacity, self.tokens + time_passed * self.fill_rate)
        self.last_time = now
        
        # Check token availability
        if tokens <= self.tokens:
            self.tokens -= tokens
            return 0  # No wait time needed
        else:
            # Calculate required wait time
            additional_tokens_needed = tokens - self.tokens
            wait_time = additional_tokens_needed / self.fill_rate
            return wait_time