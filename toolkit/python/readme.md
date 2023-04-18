```
## Singleton design
from functools import wraps

def singleton(cache_key):
  def inner_func(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
      instance = getattr(self, cache_key)
      if instance is not None:
         return instance
      instance = fn(self, *args, **kwargs)
      setattr(self, cache_key, instance)
      return instance
    return wrapper
   return inner_func
```
