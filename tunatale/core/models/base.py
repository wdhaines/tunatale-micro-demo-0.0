"""Base models and utilities for domain entities."""
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, Field, field_validator, ConfigDict, validator
from datetime import datetime
from uuid import UUID, uuid4

T = TypeVar('T', bound='BaseEntity')


class BaseEntity(BaseModel):
    """Base class for all domain entities.
    
    Attributes:
        id: Unique identifier for the entity
        created_at: Timestamp when the entity was created
        updated_at: Timestamp when the entity was last updated
    """
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('id', mode='before')
    @classmethod
    def validate_id(cls, v: Any) -> UUID:
        """Validate that the ID is a valid UUID."""
        if v is None:
            return uuid4()
            
        if isinstance(v, UUID):
            return v
            
        if isinstance(v, str):
            try:
                return UUID(v)
            except (ValueError, AttributeError, TypeError) as e:
                raise ValueError("Invalid UUID format") from e
                
        if isinstance(v, (int, bytes, bytearray)) and not isinstance(v, bool):
            try:
                return UUID(int=v)
            except (ValueError, AttributeError, TypeError) as e:
                raise ValueError("Invalid UUID format") from e
                
        raise ValueError("ID must be a valid UUID")
    
    @field_validator('created_at', 'updated_at', mode='before')
    @classmethod
    def validate_datetime(cls, v: Any, info: Any) -> datetime:
        """Validate datetime fields."""
        field_name = info.field_name
        
        if v is None:
            return datetime.utcnow()
            
        if isinstance(v, datetime):
            return v
            
        # Reject any non-string, non-numeric values
        if not isinstance(v, (int, float, str)):
            raise ValueError(
                f"{field_name} must be a datetime, timestamp, or ISO format string"
            )
            
        # Handle numeric timestamps (int or float)
        if isinstance(v, (int, float)):
            # Check if the value is a reasonable timestamp (after year 2000)
            if v < 946684800:  # 2000-01-01 00:00:00
                raise ValueError(
                    f"{field_name} timestamp is too small (before year 2000)"
                )
                
            # For very large numbers, they might be timestamps in milliseconds
            if v > 1_000_000_000_000:  # After year 2001
                v = v / 1000
                
            try:
                dt = datetime.fromtimestamp(v)
                # Ensure the resulting datetime is reasonable (not in the future + 1 year)
                current_time = datetime.utcnow().timestamp()
                max_future_time = current_time + (365 * 24 * 60 * 60)  # 1 year in the future
                if dt.timestamp() > max_future_time:
                    raise ValueError(
                        f"{field_name} is too far in the future"
                    )
                return dt
            except (ValueError, TypeError, OverflowError) as e:
                raise ValueError(
                    f"Invalid timestamp value for {field_name}: {e}"
                ) from e
                
        # Handle string values (ISO format or timestamp string)
        if isinstance(v, str):
            # First try ISO format
            try:
                return datetime.fromisoformat(v)
            except (ValueError, TypeError):
                # If not ISO format, try parsing as a timestamp string
                try:
                    timestamp = float(v)
                    # Recursively validate the timestamp
                    return cls.validate_datetime(timestamp, info)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"{field_name} must be a valid ISO 8601 datetime string or a valid timestamp"
                    ) from e
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: str
        },
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        },
        orm_mode=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    def __init__(self, **data: Any) -> None:
        """Initialize the base entity with creation timestamp."""
        # Set updated_at to created_at if not provided
        if 'updated_at' not in data and 'created_at' in data:
            data['updated_at'] = data['created_at']
        
        super().__init__(**data)
        
        # Ensure updated_at is set
        if self.updated_at is None:
            self.updated_at = self.created_at if hasattr(self, 'created_at') else datetime.utcnow()
    
    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        kwargs.setdefault('exclude_none', True)
        data = super().dict(*args, **kwargs)
        
        # Ensure ID is serialized as string
        if 'id' in data and isinstance(data['id'], UUID):
            data['id'] = str(data['id'])
            
        # Ensure timestamps are properly formatted
        for field in ['created_at', 'updated_at']:
            if field in data and isinstance(data[field], datetime):
                data[field] = data[field].isoformat()
                
        return data
    
    def __eq__(self, other: Any) -> bool:
        """Compare entities by ID only."""
        if not isinstance(other, BaseEntity):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash the entity based on its ID."""
        return hash(self.id)
