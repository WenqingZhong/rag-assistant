from typing import Dict, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    status: str = Field(..., description="healthy | unhealthy | degraded")
    message: Optional[str] = Field(None, description="Human-readable detail")


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok | degraded")
    version: str
    environment: str
    service_name: str
    services: Optional[Dict[str, ServiceStatus]] = None
