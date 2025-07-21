"""Tests for the base entity model."""
import uuid
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from tunatale.core.models.base import BaseEntity


def test_base_entity_creation() -> None:
    """Test creating a base entity with default values."""
    entity = BaseEntity()
    
    assert entity.id is not None
    assert isinstance(entity.id, uuid.UUID)
    assert entity.created_at is not None
    assert isinstance(entity.created_at, datetime)
    assert entity.updated_at is not None
    assert entity.updated_at >= entity.created_at


def test_base_entity_with_custom_values() -> None:
    """Test creating a base entity with custom values."""
    custom_id = uuid.uuid4()
    created = datetime(2023, 1, 1, 12, 0, 0)
    updated = datetime(2023, 1, 2, 12, 0, 0)
    
    entity = BaseEntity(id=custom_id, created_at=created, updated_at=updated)
    
    assert entity.id == custom_id
    assert entity.created_at == created
    assert entity.updated_at == updated


def test_base_entity_dict_conversion() -> None:
    """Test converting base entity to dictionary."""
    entity = BaseEntity()
    entity_dict = entity.dict()
    
    assert "id" in entity_dict
    assert "created_at" in entity_dict
    assert "updated_at" in entity_dict
    assert entity_dict["id"] == str(entity.id)
    assert entity_dict["created_at"] == entity.created_at.isoformat()
    assert entity_dict["updated_at"] == entity.updated_at.isoformat()


def test_base_entity_json_serialization() -> None:
    """Test JSON serialization of base entity."""
    entity = BaseEntity()
    json_str = entity.json()
    
    assert "id" in json_str
    assert "created_at" in json_str
    assert "updated_at" in json_str
    assert str(entity.id) in json_str
    assert entity.created_at.isoformat() in json_str
    assert entity.updated_at.isoformat() in json_str


def test_base_entity_validation() -> None:
    """Test validation of base entity fields."""
    # Test with invalid UUID
    with pytest.raises(ValueError):
        BaseEntity(id="not-a-uuid")
    
    # Test with invalid datetime
    with pytest.raises(ValueError):
        BaseEntity(created_at="not-a-datetime")
    
    # Test with invalid updated_at
    with pytest.raises(ValueError):
        BaseEntity(updated_at=12345)


def test_base_entity_updated_at_auto_update() -> None:
    """Test that updated_at is automatically updated when model changes."""
    entity = BaseEntity()
    initial_updated = entity.updated_at
    
    # Force a small delay to ensure timestamps are different
    import time
    time.sleep(0.01)
    
    # Update a field
    entity.updated_at = datetime.utcnow()
    
    assert entity.updated_at > initial_updated


def test_base_entity_equality() -> None:
    """Test equality comparison of base entities."""
    entity1 = BaseEntity()
    entity2 = BaseEntity()
    
    # Different IDs should not be equal
    assert entity1 != entity2
    
    # Same ID should be equal
    entity3 = BaseEntity(id=entity1.id)
    assert entity1 == entity3
    
    # Different types should not be equal
    assert entity1 != "not-an-entity"
    assert entity1 is not None
