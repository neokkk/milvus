try:
    import _ctypes
    print("The _ctypes module is available")
except ModuleNotFoundError:
    print("The _ctypes module is not available")
