# Efficient-Semantic-Guidance-High-resolution-Video-Matting

Video matting has made significant progress in trimap-based field. However, researchers are increasingly interested in auxiliary-free matting because it is more useful in real-world applications. We propose a new efficient semantic-guidance high-resolution video matting network for human body. This network maintains efficiency while improving the comprehension of semantic feature. We still apply the convolutional network as the backbone while also employing the transformer in the encoder. The transformer is used as a submodule to provide semantic features to help the convolutional network, while ensuring that the network is not overly bloated. In addition, a channel-wise attention mechanism is introduced in the decoder to improve the representation of semantic feature. In comparison to the current state-of-the-art methods, the method proposed in this paper achieves better results while maintaining the speed and efficiency of prediction. We can complete the real-time auxiliary-free matting for high-resolution video (4K or HD).

# Requirment

