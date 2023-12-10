# denoiserGAN

## Abstract
This final project implements a Generative Adversarial Network, better known as GAN, for image denoising on the EMNIST dataset. A U-NET convolutional architecture is applied as the generator to remove image noise and a discriminator outputs the confidence it has that the reconstructed image is the cleaned image. Also, classical image denoising methods such as Gaussian Blur, Median Filtering, and Weiner Filtering are compared to our GAN-based method to determine if it makes sense to transition to this deep learning technique for denoising.

## Introduction
In the rapidly evolving field of image processing, the quest for efficient denoising techniques remains a pivotal challenge. This final project implements an unorthodox approach to image denoising, using Generative Adversarial Networks (GANs), specifically on the EMNIST dataset. Created by Goodfellow et al. in 2014, GANs have revolutionized the unsupervised machine learning field with their dual-component structure of a generator and a discriminator. These components work in tandem, with the generator producing realistic images from noise, and the discriminator differentiating between genuine and generated images.
My project uses a U-NET convolutional architecture as the generator. This design choice is intentional, as it enables the removal of noise while preserving the integrity of the original image. Concurrently, the discriminator assesses the authenticity of the reconstructed images, thus ensuring the effectiveness of the denoising process. This dual mechanism enhances the quality and detail preservation and adapts to complex noise patterns, thereby offering a contextual understanding of the images.
To validate my GAN-based approach, I compare it with the classical image-denoising methods of Gaussian Blur, Median Filtering, and Weiner Filtering. This comparison is crucial to judge whether the transition to deep learning techniques, despite their complexity and resource-intensive nature, is justified for image-denoising tasks.
The adaptability and versatility of GANs in handling diverse and complex noise patterns can make them a valid benchmark in the image-denoising field. However, the inherent complexities, such as the need for extensive training and tuning, along with the unpredictability and potential for introducing unrealistic features, are challenges that justify further improvement. My project aims not only to demonstrate the practical application of GANs in image denoising but also to contribute to the broader discourse on the feasibility and optimization of deep learning techniques in image processing.

## Future Work
In the future to expand upon this, I could….
1.	Implement a conditional GAN (cGAN) that can denoise images based on specific noise types or levels
2.	Explore techniques for progressive GAN training to improve the denoising quality
3.	Investigate the impact of dataset size and diversity on GAN-based image denoising
