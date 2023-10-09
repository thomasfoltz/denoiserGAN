# denoiserGAN
GAN-based image denoiser

## Project Description
In this project, I will explore the application of Generative Adversarial Networks (GANs) for image denoising. GANs are known for their ability to generate realistic data, and they can be adapted for noise reduction tasks. I will develop and train a model to denoise noisy EMNIST dataset images using a GAN-based approach.

## Project Rationale
1.	I’ll use the MNIST dataset with added image noise to test on
2.	Then I could process these noisy images, including resizing and normalizing them
3.	Then I can design and implement a GAN architecture for image denoising. The GAN should consist of a generator and a discriminator. The generator's task is to remove noise from input images, while the discriminator distinguishes between denoised images and real clean images
4.	After building it, I can apply the GAN on the noisy image dataset. The generator should learn to produce denoised images that are visually similar to the clean images in the dataset
5.	I will evaluate the GAN's performance on a separate testing dataset, measuring the quality of the denoised images by comparing the results to other denoising methods and visualizing the denoising results by comparing noisy input images with the GAN's denoised outputs.
6.	Then I could fine tune with different GAN architectures, loss functions, and hyperparameters to optimize the denoising performance

## Future Work
In the future to expand upon this, I could….
1.	Implement a conditional GAN (cGAN) that can denoise images based on specific noise types or levels
2.	Explore techniques for progressive GAN training to improve the denoising quality
3.	Investigate the impact of dataset size and diversity on GAN-based image denoising
