# LLMit: Easily call Large Language Models from a unified API

LLMit is a Python library for easily calling large language models from a unified API.

## Supported Large Language Models

From OpenAI:
- GPT-4 (Base, 32K) (Chat Engine)
- GPT-3.5 (ChatGPT) (Chat Engine)
- GPT-3 (Davinci (v2,v3), Ada, Babbage, Curie) (Completion Engine)

From Anthropic:
- Claude (Base, 100K) (Chat Engine)
- Claude Instant (Base, 100K) (Chat Engine)

From Google/GCP:
- PaLM (Chat and Completion Engines)

From Huggingface
- GPT-2 (Base, Medium, Large, XL) (Completion Engine)
- GPT-Neo (125M, 1.3B, 2.7B) (Completion Engine)
- GPT-J (6B) (Completion Engine)

From Facebook (via Huggingface)
- Llama (7B, 13B, 30B, 65B) (Completion Engine)
- OPT (66B) (Completion Engine)

From Stanford (via Huggingface)
- Alpaca (7B) (Chat Engine)

From Berkeley (via Huggingface)
- Koala (7B, 13B_v1, 13B_v2) (Chat Engine)
- Vicuna (7B, 13B) (Chat Engine)

From StabilityAI (via Huggingface)
- StableLM (7B, 13B) (Chat and Completion Engines)


## Installation

Llmit can easily be installed using pip:
```bash
pip install llmit
```

Each of the LLMs may need additional setup, which you can find in the engine setup section below.



## Usage

For completion engines, you can use the `LLMEngine` class:
```python
from llmit import LLMEngine

gpt2 = LLMEngine.from_string("gpt2")
completion = gpt2("I enjoy walking with my cute dog, but sometimes he gets scared and")
print(completion)
```

For chat engines, you can use the `LLMChat` class:
```python
from llmit import LLMChat, Conversation, Speaker

conversation = Conversation()
conversation.add_turn("You are a funny person.", speaker=Speaker.SYSTEM)
conversation.add_turn("Hi, how are you?", speaker=Speaker.USER)
conversation.add_turn("I am doing well, how about you?", speaker=Speaker.AI)
conversation.add_turn("What are you planning to do today?", speaker=Speaker.USER)

gpt4 = LLMChat.from_string("gpt4")
next_turn = gpt4(conversation)

print(next_turn)
```


## Individual Engine Setup

Each engine may require some specific details to be passed in. For example, OpenAI engines require an API key. These
details are generally set up with environment variables.

### OpenAI Engines

For OpenAI engines, you will need to set the `OPENAI_API_KEY` and `OPENAI_API_ORG` environment variables. You can find
your API key and organization ID on the [OpenAI dashboard](https://platform.openai.com/). You can set these environment
variables in your shell or in a `.env` file in the root of your project. For example, in a `.env` file, you would have:
```bash
OPENAI_API_KEY=<your key>
OPENAI_API_ORG=<your org id>
```
or on the command line:
```bash
export OPENAI_API_KEY=<your key>
export OPENAI_API_ORG=<your org id>
```

### Anthropic Engines
For Anthropic engines, you will need to set the `ANTHROPIC_API_KEY` environment variable. You can find your API key at
the [Anthropic dashboard](https://console.anthropic.com/account/keys). You can set this environment variable in
your shell or in a `.env` file in the root of your project. For example, in a `.env` file, you would have:
```bash
ANTHROPIC_API_KEY=<your key>
```
or on the command line:
```bash
export ANTHROPIC_API_KEY=<your key>
```

### Vertex Engines (PaLM)
For Google engines, we use the Vertex cloud API, which requires a Google Cloud Platform (GCP) project. You can create a
GCP project at the [GCP console](https://console.cloud.google.com/). You will also need to enable the Vertex AI API for
your project, set up a services account, and download the account JSON credentials. You can find instructions for that
following steps 1 to 6 of the tutorial  [here](https://cloud.google.com/vertex-ai/docs/tutorials/image-recognition-automl).
Finally, you will need to set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the JSON file.
You can set this environment variable in your shell or in a `.env` file in the root of your project. For example, in a
`.env` file, you would have:
```bash
GOOGLE_APPLICATION_CREDENTIALS=<path to your JSON file>
```
or on the command line:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=<path to your JSON file>
```

### Bard
For the Bard engine, you will need to get your Bard SESSION_ID.  Get the value of this variable by first going to
https://bard.google.com/, then log in, press F12 for console, and go to the "Application" tab, then "Cookies",
then copy the value of the "__Secure-1PSID" cookie. You can then set the environment variable:
```bash
GOOGLE_BARD_SESSION_ID=<your session id>
```

### Huggingface Engines
Most of the huggingface engines require no additional setup, however, some of the larger models require a GPU to run
with any kind of efficiency (and some require multiple GPUs with large amounts of memory). You can find more details
about the requirements for each model on the [Huggingface model hub](https://huggingface.co/models).

### Llama, Alpaca, Koala, and Vicuna Engines
For these engines, you will need to obtain and postprocess the weights yourself (due to Facebook's licensing). You can
find the instructions for doing so on each model page:
- Llama: https://huggingface.co/docs/transformers/main/model_doc/llama
- Alpaca: https://github.com/tatsu-lab/stanford_alpaca#recovering-alpaca-weights
- Koala: https://github.com/young-geng/EasyLM/blob/main/docs/koala.md
- Vicuna: https://github.com/lm-sys/FastChat#vicuna-weights

Once the weights have been downloaded and processed, you can set the following environment variables to the root
directory containing a folder for each variant (for example, `root_dir/llama_7B/weights.bin`, the root directory would
be `root_dir`):
```bash
LLAMA_WEIGHTS_ROOT=<path to the llama weights>
ALPACA_WEIGHTS_ROOT=<path to the alpaca weights>
KOALA_WEIGHTS_ROOT=<path to the koala weights>
VICUNA_WEIGHTS_ROOT=<path to the vicuna weights>
```

## Citation

If you use LLMit in your work, please cite:

```
@misc{LLMit,
  author = {David Chan},
  title = {LLMit: Easily call Large Language Models from a unified API},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{
    https://github.com/DavidMChan/llmit
  }}
}
```

## License

LLMit is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for more information.
