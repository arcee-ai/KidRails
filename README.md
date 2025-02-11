# KidSmart: Replicating a Child-Safe LLM Model

This repository provides the code, data references, and instructions necessary to replicate our KidSmart model—an open-source, child-safe large language model (LLM) built on the KidRails framework. The project was developed to set a new industry standard for age-appropriate, transparent AI interactions for children, and to promote collaboration among educators, developers, and researchers.

## Table of Contents
- Overview
- Project Motivation
- Model & Training Process
- License
- Acknowledgments

## Overview

KidSmart is an LLM adapted specifically for ensuring safe, age-appropriate interactions with children. By leveraging a model-agnostic, open-source approach, our framework builds upon robust safety mechanisms and transparent training processes. The repository includes:
- Code for training and fine-tuning the LLM (based on Llama 3.1 8B) for child-safe responses.
- Data processing scripts to handle our training dataset, which includes curated Golden QA pairs from teachers and educators.
- Utilities for evaluating model performance against defined safety and age-appropriateness metrics.

## Project Motivation

As LLMs become increasingly integrated into digital experiences for children, ensuring their safety is paramount. Our key motivations were:
- **Choosing Llama 3.1:**  
  Our team conducted extensive testing across various models. Llama 3.1 was selected for its balance of performance and efficiency, complemented by Arcee AI’s expertise in developing smaller yet highly effective models.
- **Open-Source Commitment:**  
  Making KidSmart open-source was crucial to set a new benchmark in child-safe AI. By sharing all training data and methods, we empower developers and researchers worldwide to audit, contribute, and adapt the model for diverse applications.
- **Robust Safety Measures:**  
  The model incorporates real Golden QA pairs sourced from educators to ensure responses are both safe and encouraging for children. Additionally, we employed a novel “spectrum” training process to identify and emphasize layers with the highest signal-to-noise ratio, thereby improving computational efficiency and overall response quality.

## Model & Training Process

The training of KidSmart involved several key steps:
1. **Data Collection:**
   - We collaborated with the AngelKids team to gather real Golden QA pairs provided by teachers and educators.
   - These pairs include examples of both safe and unsafe responses, serving as the basis for aligning the model’s outputs with child safety guidelines.
2. **Data Generation & Augmentation:**
   - Using our curated dataset, we generated a variety of safe versus non-safe responses.
   - The dataset was further enriched with diverse scenarios to cover a wide range of topics relevant to children’s interactions.
3. **Spectrum Analysis in Training:**
   - A spectrum-based method was implemented to analyze the signal quality across different model layers.
   - Layers contributing the most significant signal-to-noise ratio were prioritized, which in turn improved computational efficiency and model performance.
4. **Fine-Tuning on Llama 3.1 8B:**
   - The base model, Llama 3.1 8B, was chosen after rigorous testing for its optimal performance.
   - Fine-tuning was performed using our custom dataset and training routines, incorporating both the Golden QA pairs and the spectrum analysis outcomes.

For further technical details and the rationale behind each step, please refer to the KidRails white paper included in the repository.

## License

This project is open-sourced under the MIT License. We encourage academic, educational, and research use, as well as commercial applications that align with our mission of promoting safe digital experiences for children.

## Acknowledgments

- **AngelKids Team:** For providing the invaluable Golden QA pairs and insights from educators.
- **Arcee AI:** For their expertise in developing and refining small language models.
- **Community Contributors:** For ongoing contributions that help set the standard for child-safe AI.

For more information about our mission and to explore additional resources, please visit [AngelKids.ai](https://www.angelq.ai/) and [Arcee.ai](https://www.arcee.ai/).

This README documents our comprehensive approach to training a child-safe LLM model. By replicating this process, you help advance the broader goal of creating transparent, safe, and engaging AI experiences for children.