const Tensor = @import("zml/core").Tensor;

// LeakyRelu ONNX, alpha 0,2. La passe de fusion cuDNN de XLA reconnaît le Select
// sur le produit négatif lorsque la convolution voisine peut fournir l'activation.
pub fn leaky(x: Tensor, use_select: bool) Tensor {
    if (!use_select) return x.leakyReLU(0.2);
    return x.cmp(.GT, .scalar(0, x.dtype())).select(x, x.scale(0.2));
}
