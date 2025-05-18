DTYPE = 'float32'   # Default data type for numerical operations. 
                    # Using 'float32' can be faster on some hardware

USE_GPU = True      # Set to True to use GPU for computations if available

assert DTYPE in ['float32', 'float64'], "DTYPE must be either 'float32' or 'float64'"
