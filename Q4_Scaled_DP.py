import numpy as np

def scaled_dot_product_attention(Q, K, V):
    # Step 1: Dot product of Q and K transpose
    matmul_qk = np.matmul(Q, K.T)

    # Step 2: Scale by sqrt(d)
    d_k = K.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    # Step 3: Softmax to get attention weights
    attention_weights = np.exp(scaled_attention_logits)
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

    # Step 4: Multiply weights with V
    output = np.matmul(attention_weights, V)

    return output, attention_weights

# Test inputs
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("Attention Weights:\n", attention_weights)
print("\nOutput:\n", output)