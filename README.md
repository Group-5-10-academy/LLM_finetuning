# LLM_finetuning
Welcome to the LLM Finetuning project! This project aims to enhance Natural Language Processing (NLP) capabilities for Amharic language by fine-tuning language models for quality embedding and text generation. The project addresses the lack of extensive, high-quality text and audio datasets for these languages, enabling various NLP applications such as semantic search, content generation, chatbot support, sentiment analysis, and speech recognition.
Enabling Quality Embedding and Text Generation for Amharic Language

We finetuned Amharic language pretrained model from GARI Logstics.

<h3>Instructions</h3>
1.Accept Llama2 license on huggingface and download it like this:
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf [if you don't have access to meta-llama replace with https://huggingface.co/NousResearch/Nous-Hermes-llama-2-7b]
 2. Download the amharic finetune from huggingface like this:
git lfs install
git clone https://huggingface.co/iocuydi/llama-2-amharic-3784m
3. Clone this github repository
4. Then inside inference/run_inf.py:
comment the import safety_utils line
change the MAIN_PATH to the path to folder you downloaded from step 1
change the peft_model to the path you cloned in the step 2
Go to your llama2 folder(from step 1) and replace the tokenizer.model file with the one you find from the 2nd step
set quanitzation=True inside the main function before the load_model function call
5. Finally run the inference/run_inf.py file