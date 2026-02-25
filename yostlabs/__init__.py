"""
Yost Labs namespace package.

This package uses pkgutil-style namespace package to allow multiple
distributions to provide modules under the yostlabs namespace.
"""

__path__ = __import__('pkgutil').extend_path(__path__, __name__)
