"""
Transform Registry Pattern Implementation

Provides a centralized registry for dynamically loading and creating
transformation classes without hardcoded conditional logic.
"""
from typing import Dict, Type, List
from transformations.base_transform import BaseTransform


class TransformRegistry:
    """Registry for dynamically loading transformation classes"""
    
    def __init__(self):
        self._transforms: Dict[str, Type[BaseTransform]] = {}
        self._initialized = False
    
    def register(self, name: str, transform_class: Type[BaseTransform]):
        """Register a transformation class
        
        Args:
            name: String identifier for the transformation
            transform_class: Class that inherits from BaseTransform
        
        Raises:
            ValueError: If transform_class doesn't inherit from BaseTransform
            ValueError: If name is already registered
        """
        # Validation
        if not issubclass(transform_class, BaseTransform):
            raise ValueError(f"Transform class {transform_class.__name__} must inherit from BaseTransform")
        
        if name in self._transforms:
            existing_class = self._transforms[name].__name__
            raise ValueError(f"Transform '{name}' already registered with class {existing_class}")
        
        self._transforms[name] = transform_class
        print(f"Registered transformation: '{name}' -> {transform_class.__name__}")
    
    def create(self, name: str, **params) -> BaseTransform:
        """Create transformation instance
        
        Args:
            name: String identifier for the transformation
            **params: Parameters to pass to the transformation constructor
        
        Returns:
            Configured transformation instance
        
        Raises:
            ValueError: If transformation name is not registered
        """
        if name not in self._transforms:
            available = list(self._transforms.keys())
            raise ValueError(f"Unknown transformation '{name}'. Available: {available}")
        
        transform_class = self._transforms[name]
        try:
            return transform_class(**params)
        except Exception as e:
            raise ValueError(f"Error creating transformation '{name}': {str(e)}")
    
    def list_available(self) -> List[str]:
        """List all registered transformation names"""
        return list(self._transforms.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if a transformation is registered"""
        return name in self._transforms
    
    def get_class(self, name: str) -> Type[BaseTransform]:
        """Get the transformation class (not instance)"""
        if name not in self._transforms:
            available = list(self._transforms.keys())
            raise ValueError(f"Unknown transformation '{name}'. Available: {available}")
        return self._transforms[name]
    
    def unregister(self, name: str):
        """Remove a transformation from registry (useful for testing)"""
        if name in self._transforms:
            del self._transforms[name]
            print(f"Unregistered transformation: '{name}'")
    
    def clear(self):
        """Clear all registered transformations (useful for testing)"""
        self._transforms.clear()
        print("Cleared all registered transformations")


# Global registry instance
_global_registry = TransformRegistry()


def get_registry() -> TransformRegistry:
    """Get the global transformation registry"""
    global _global_registry
    
    # Auto-initialize with default transformations on first access
    if not _global_registry._initialized:
        _initialize_default_transforms()
        _global_registry._initialized = True
    
    return _global_registry


def _initialize_default_transforms():
    """Initialize registry with default transformations"""
    from transformations.log_transform import LogTransform
    from transformations.first_difference import FirstDifference
    from transformations.box_cox_transform import BoxCoxTransform
    
    registry = _global_registry
    registry.register('log_transform', LogTransform)
    registry.register('first_difference', FirstDifference)
    registry.register('box_cox_transform', BoxCoxTransform)
    
    print("Initialized transform registry with default transformations")


# Convenience functions
def register_transform(name: str, transform_class: Type[BaseTransform]):
    """Register a transformation in the global registry"""
    get_registry().register(name, transform_class)


def create_transform(name: str, **params) -> BaseTransform:
    """Create a transformation from the global registry"""
    return get_registry().create(name, **params)


def list_transforms() -> List[str]:
    """List all available transformations in the global registry"""
    return get_registry().list_available() 