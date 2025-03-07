# Adversarial Patch Generation for ResNet34

## Overview

This project implements the generation of adversarial patches designed to fool a pre-trained ResNet34 model, as part of the AIPI 590: Emerging Trends in AI course at Duke University. The objective is to create a physical patch that, when applied to an image, causes the model to misclassify it. This repository contains the code, documentation, and results of this project, drawing inspiration from the work of Goodfellow et al. (2015) and Metzen et al. (2017) on adversarial examples and perturbations.

## Project Goals

-   Design and implement an effective adversarial patch capable of fooling a pre-trained ResNet34 image classification model.
-   Leverage a class from the ImageNet dataset to create the patch.
-   Incorporate a creative element to enhance its functionality or presentation (e.g., "disguising" the patch in a sticker).
-   Produce a physical patch that successfully alters the model's predictions in a live test.
-   Document the code and methodology clearly.
-   Showcase creative application of adversarial techniques.

## Setup and Installation

### Prerequisites

The project requires several Python packages which are listed in `requirements.txt`.

1.  **Clone the repository:**

    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```

2.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, you can run the following commands directly in your Google Colab notebook:

    ```bash
    !pip install gdown
    ```

3.  **Download necessary files using gdown:**

    ```python
    import gdown

    reqtxt_file_id = "1U4V_9N114msCNDg139MpMfEDtgG-xVko"
    reqtxt_output_file = "requirements.txt"

    gdown.download(f"[https://drive.google.com/uc?id=](https://drive.google.com/uc?id=){reqtxt_file_id}", output=reqtxt_output_file)
    ```

## Implementation Steps

1.  **Open the Google Colab notebook (`adversarial-attack_adversarial-patchgeneration.ipynb`).**
2.  **Run the notebook cells sequentially.** The notebook includes:
    -   Downloading and installing necessary packages.
    -   Loading the pre-trained ResNet34 model.
    -   Implementing the adversarial patch generation algorithm.
    -   Visualizing the generated patches.
    -   Saving the patches for physical application.
3.  **To generate your own patch:**
    -   Modify the target class name in the notebook.
    -   Adjust the patch size and other parameters as desired.
    -   Run the notebook to generate the new patch.
4.  **Print the generated patch:**
    -   Ensure the patch is printed in color if color is a critical component of its effectiveness.
    -   Apply the patch to a physical object or image.
5.  **Test the patch:**
    -   Use the provided Streamlit app (`https://resnet34-classifier.streamlit.app/`) or a similar setup to test the physical patch with the ResNet34 model.

## Code Explanation

The notebook utilizes PyTorch to generate adversarial patches using gradient-based methods. Key components include:

-   **Model Loading:** Loading the pre-trained ResNet34 model from Torchvision.
-   **Patch Generation:** Implementing an iterative algorithm to generate a patch that maximizes the misclassification probability.
-   **Visualization:** Displaying the generated patches using Matplotlib.
-   **Creative Implementation:** The notebook includes an example of "disguising" the patch within a "star" shape, demonstrating the creative aspect of the assignment.

## Results

The notebook includes an example of a generated adversarial patch for the "tree frog" class. The patch is designed to cause the ResNet34 model to misclassify images when the patch is applied.

## Creative Component

This project includes a creative component where the adversarial patch is designed as a "star print". This demonstrates how adversarial patches can be integrated into real-world scenarios in a visually appealing and potentially inconspicuous manner.

## Conclusion

This project demonstrates the vulnerability of deep learning models to adversarial attacks. By generating and applying adversarial patches, we can manipulate the predictions of a ResNet34 model. This project highlights the importance of developing robust AI models that are resilient to such attacks. The work of Goodfellow et al. (2015) and Metzen et al. (2017) underscores the broader implications of adversarial examples and their potential impact on machine learning systems.

## Future Extensions

-   Explore different adversarial patch generation techniques.
-   Test the robustness of the patches against various defense mechanisms.
-   Investigate the transferability of adversarial patches across different models.
-   Implement real-world scenarios to demonstrate the potential risks of adversarial patches.

## Requirements

See `requirements.txt` for complete list of dependencies.

## References

-   [Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." ICLR 2015.](https://arxiv.org/abs/1412.6572)

-   [Hendrik Metzen, Jan, et al. "Universal adversarial perturbations against semantic image segmentation." Proceedings of the IEEE International Conference on Computer Vision. 2017.](https://openaccess.thecvf.com/content_ICCV_2017/papers/Metzen_Universal_Adversarial_Perturbations_ICCV_2017_paper.pdf)

-   [Anant Jain. "Breaking neural networks with adversarial attacks." Blog post 2019.](https://www.anantjain.dev/posts/the-intuition-behind-adversarial-attacks-on-neural-networks)

-   [Dr. Brinnae's Notebook Link](https://github.com/AIPI-590-XAI/Duke-AI-XAI/blob/main/adversarial-ai-example-notebooks/adversarial_attacks_patches.ipynb)

-   [GeminiAI](https://gemini.google.com/app)

-   [ClaudeAI](https://claude.ai/new)

---

📚 **Author of Notebook:** Michael Dankwah Agyeman-Prempeh [MEng. DTI '25]