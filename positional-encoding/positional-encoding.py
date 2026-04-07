import numpy as np
import math

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    PE = np.zeros((seq_len, d_model))

    for i in range(seq_len):
        for j in range(d_model):
            if j%2==0:
                # sine
                PE[i][j] = math.sin((i)/(base**(j/d_model)))
            else:
                # cosine
                PE[i][j] = math.cos((i)/(base**((j-1)/d_model)))

    return PE