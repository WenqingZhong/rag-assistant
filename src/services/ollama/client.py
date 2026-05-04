import logging
from typing import Any, Dict, List, Optional

import httpx

from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with the Ollama local LLM service."""

    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(float(timeout))

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama is healthy and return version info.

        Returns {"status": "healthy", "message": ..., "version": ...}
        Raises OllamaConnectionError / OllamaTimeoutError on failure.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/version")
                if response.status_code == 200:
                    version_data = response.json()
                    return {
                        "status": "healthy",
                        "message": "Ollama service is running",
                        "version": version_data.get("version", "unknown"),
                    }
                raise OllamaException(f"Ollama returned status {response.status_code}")
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Ollama health check failed: {e}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """Return list of locally available models from /api/tags."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    return response.json().get("models", [])
                raise OllamaException(f"Failed to list models: {response.status_code}")
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error listing models: {e}")

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate text using the specified model.

        Args:
            model:  Model name (e.g. "llama3.2:1b")
            prompt: Input prompt
            stream: Streaming not yet implemented — keep False
            **kwargs: Extra Ollama generation parameters (temperature, etc.)
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {"model": model, "prompt": prompt, "stream": stream, **kwargs}
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                if response.status_code == 200:
                    return response.json()
                raise OllamaException(f"Generation failed: {response.status_code}")
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error generating with Ollama: {e}")
