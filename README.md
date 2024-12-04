In the first part of this project, we conduct a comprehensive analysis of the Mamba model and its key innovations, highlighting its major technical contributions and the improvements it introduces over existing approaches. We examine how the Mamba model addresses current challenges in sequence modeling by incorporating features like the selection mechanism and hardware-aware algorithms, which enhance its efficiency, scalability, and overall performance.

Following this exploration, we propose three potential directions for further extending the Mamba model's capabilities. Each extension builds on the model's foundational principles, aiming to improve its performance, broaden its applicability, or address specific limitations. To demonstrate the feasibility of these extensions, we present preliminary results from one of them, offering valuable insights into the potential impact of the proposed ideas. These initial findings lay the groundwork for future research and development, highlighting Mamba’s potential to drive advancements in the field of deep learning.


# To run the experiments
```python train.py --dim 32 --depth 4
   python train.py --dim 64 --depth 8
   python train.py --dim 32 --depth 16
```

In the second part of the project, we conduct a thorough examination of the VILA model and its extensions. Similarly to the report on Mamba, the comprehensive study will review the core characteristics of VILA, identify significant technical innovations, and potential areas of improvement. As such, our goal is to provide a detailed summary of the VILA model's contributions and discuss how this model outperforms state-of-the-art Visual Language Models (VLMs) such as LLaVA-1.5.

Then, we analyze an efficiency bottleneck inherent to VLMs and the VILA model architecture, and propose a novel approach designed to mitigate this limitation. Specifically, the time complexity introduced by the self-attention mechanism in the ViT architecture, which forms the basis of Transformer encoder layers in VLMs, scales quadratically with input length. To mitigate this, we propose the implementation of a Dynamic Depth Visual Encoding model aimed at reducing this complexity. To validate the performance of this model, we compare it against a baseline visual encoding model using the MNIST dataset. Both the testing accuracy and floating-point operations per second are recorded. These findings and improvements could serve as inspiration for further work on VILA, and provide a foundation for developing more efficient and robust Visual Language Models in the future. The code comparing both a baseline ViT model and the DD ViT model can be found in the DynamicDepthVisualEncoding.ipynb file.

# To run the experiment (baseline vs DD ViT)
```Simply run the entire DynamicDepthVisualEncoding.ipynb file```

