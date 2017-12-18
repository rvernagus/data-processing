import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

def fillna(values):
    """Fills na values in the supplied DataFrame.

    Uses 'mean' strategy by default.
    >>> fillna(np.array([[1,2,3],[4,np.nan,6],[np.nan,8,9]]))
    array([[ 1. ,  2. ,  3. ],
           [ 4. ,  5. ,  6. ],
           [ 8.5,  8. ,  9. ]])

    na values can be np.nan or None.
    >>> fillna(np.array([[1,2,3],[4,None,6],[None,8,9]]))
    array([[ 1. ,  2. ,  3. ],
           [ 4. ,  5. ,  6. ],
           [ 8.5,  8. ,  9. ]])

    It will convert strings to numbers if it can.
    >>> fillna([[None,'2','3'],['4',None,'6'],['7','8',None]])
    array([[ 2.5,  2. ,  3. ],
           [ 4. ,  5. ,  6. ],
           [ 7. ,  8. ,  7.5]])
    """
    imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
    imp.fit(values)
    return imp.transform(values)

if __name__ == "__main__":
    import doctest
    doctest.testmod()