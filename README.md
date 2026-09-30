# Conversation block ranker

This command-line tool reads a UTF-8 conversation text file, splits it into overlapping blocks, asks a local OpenJev server a set of yes/no questions about each block, and writes the 100 blocks with the highest yes-probability for each question.

## Privacy and models

The ranker sends requests only to literal loopback IPs (`127.0.0.1` or `::1`) and rejects other hosts. It disables system HTTP proxies and refuses redirects, so conversation requests cannot be forwarded to a LAN or internet host. It has no cloud API integration, telemetry, or model download code. Conversation text and results stay on the PC when OpenJev is bound to loopback.

OpenJev runs as a separate local server and must be available at `http://127.0.0.1:8080`. The default model is `verdict-1.4` (151M parameters, up to 512 input tokens), a practical starting point for a modest PC. `laya-1.0` is larger (421M parameters) but accepts up to 1,024 tokens. Set the block size conservatively for the model selected; the default is 200 words. Model names and availability depend on the OpenJev server configuration.

The upstream OpenJev server may download software and model weights during setup or its first start. Those are downloads; the ranker never uploads conversation text. For a machine that must remain offline, install the software and cache the weights beforehand, then configure OpenJev to use those local files. Bind the OpenJev server to loopback; the ranker itself will not send text to a LAN or internet host.

On Windows, install Docker Desktop and use Linux containers. The launcher builds the upstream OpenJev Docker base and selected small-model image, then binds the server to `127.0.0.1:8080`. You do not need to install or configure an Ubuntu distribution yourself. Docker Desktop may use WSL2 internally; NVIDIA GPU access from Docker Desktop on Windows requires WSL2 GPU support and a compatible NVIDIA driver.

To run Verdict manually instead of using the launcher, build the upstream images and start the container:

```powershell
git clone https://github.com/razorback16/openjev
cd openjev
docker build -f docker/Dockerfile.base -t razorback16/openjev-base:cu130-torch2.13 .
docker build --build-arg BASE_IMAGE=razorback16/openjev-base:cu130-torch2.13 `
  -f docker/Dockerfile.verdict -t openjev-verdict .
docker volume create openjev-huggingface
docker run --rm --name openjev-verdict -p 127.0.0.1:8080:8080 `
  -v openjev-huggingface:/root/.cache/huggingface `
  -e OPENJEV_DEVICE=cpu openjev-verdict
```

For CUDA, add `--gpus all` to `docker run` and set `OPENJEV_DEVICE=cuda`. The first image build and model start download dependencies and weights; the named Docker volume keeps model weights between runs. The ranker requires Python 3.10 or newer and has no third-party Python dependencies.

## Use

Pass questions directly, repeating `--question` as needed:

```powershell
python .\conversation_ranker.py .\conversation.txt `
  --question "Does this block discuss politics?" `
  --question "Does this block mention vision?"
```

Or store one question per line in a UTF-8 file:

```powershell
python .\conversation_ranker.py .\conversation.txt --questions-file .\questions.txt
```

The default output is `top_blocks.json`. Each question maps to up to 100 results, sorted by descending `yes_probability`, with the source block number and full block text included.

Useful options:

```text
--model verdict-1.4       Select a model served by OpenJev
--block-words 200        Maximum words in each block
--overlap-words 30       Shared words between adjacent blocks
--top 100                Results per question
--output results.json    Output path
--base-url URL           Local OpenJev URL (loopback only)
```

For example, use `--model laya-1.0 --block-words 350` when that model is enabled locally. The tool checks the configured model against `GET /v1/models` before scoring. If the local server requires a client key, set `OPENJEV_API_KEY` in the environment; the key is sent only to the loopback server.

The probability is OpenJev's estimated probability that the answer is “yes,” not a guarantee of truth. Review ranked blocks before relying on them.

## Simple launcher

For the interactive flow, run:

```powershell
python .\conversation_app.py
```

Choose Verdict or Laya, enter a conversation file path, then enter questions one at a time. Submit a blank question line when finished. Results are written beside the conversation file; enter another path to continue or `Q` to close. The launcher downloads the OpenJev repository into `%LOCALAPPDATA%\ConversationBlockRanker\openjev` if it is missing, builds the shared Docker base and selected backend image, creates a persistent Docker volume for model weights, starts the server, and stops its container when you close the app. Built images and cached weights remain available for the next run. A matching server already running on localhost is reused and left running when the app closes.

The launcher offers these OpenJev-compatible models. Memory figures are practical recommendations for running the app, not exact model-weight sizes. System RAM is separate from GPU VRAM.

| Model | Input limit | System RAM | GPU VRAM | Notes |
| --- | ---: | ---: | ---: | --- |
| Verdict 151M | 512 tokens | 16 GB | 2 GB | Runs on CPU or NVIDIA GPU |
| Laya 421M | 1,024 tokens | 16 GB | 4 GB | Runs on CPU or NVIDIA GPU; recommended for 8 GB cards such as the RTX 2080 SUPER |
| JevK5 4B | 16,384 tokens | 32 GB | 12 GB | NVIDIA GPU; a practical fit for desktop RTX 5070 |
| CLM 8B | 2,048 tokens | 32 GB | 12 GB | NVIDIA GPU; Qwen3-8B-based, practical on a 12 GB card at its configured context limit |
| DiffusionGemma 26B-A4B | 65,536 tokens | 64 GB | 24 GB minimum | NVIDIA GPU; does not fit on an RTX 5070 with 12 GB VRAM |

For an RTX 2080 SUPER with 8 GB VRAM, Laya is the recommended GPU choice in this menu. The vLLM-based JevK5 and CLM backends need about 12 GB, and DiffusionGemma needs at least 24 GB. The 5070 desktop card has 12 GB VRAM, making JevK5 or CLM the largest practical choices in this menu; the laptop RTX 5070 has less VRAM, so use Verdict or Laya unless your specific configuration has enough GPU memory. CPU mode is available only for Verdict and Laya. These figures assume one model at a time; larger contexts or concurrent requests can need more memory.

The general-purpose Qwen3 14B and 32B models are not direct choices here: this ranker expects OpenJev's typed yes/no scoring API, while those Qwen3 models are general text generators. Running them from this app would require a separate inference and scoring integration. If OpenJev is already running, the launcher also lists every additional model returned by its local `/v1/models` endpoint.

Before starting a GPU model, the launcher verifies that Docker can access the GPU and that PyTorch CUDA is usable. Docker Desktop GPU support, WSL2 GPU support, and a compatible NVIDIA Windows driver are required. The first Docker image build can take a while and use several GB of disk space.

GPU models reserve a fraction of free VRAM for vLLM (85-90% by default). If startup fails with a message like "Free memory on device cuda:0 ... is less than desired GPU memory utilization", another process is using VRAM, or the card is close to the model's minimum. Close other GPU applications, or set `OPENJEV_GPU_UTIL` (for example `0.6`) in the environment before starting the launcher to lower vLLM's reserved fraction.

JevK5 can use system RAM for some model weights through vLLM CPU offload, at a substantial speed cost. For an 8 GB GPU, try this PowerShell configuration before launching; it offloads 4 GB and limits the context to 4,096 tokens:

```powershell
$env:OPENJEV_GPU_UTIL = "0.65"
$env:OPENJEV_MAX_MODEL_LEN = "4096"
$env:OPENJEV_VLLM_ARGS = "--cpu-offload-gb 4"
python .\conversation_app.py
```

CPU offload uses system RAM, not the GPU's shared-memory pool. JevK5 is not listed as an 8 GB fit, so this configuration is best-effort and requires enough free system RAM; Verdict or Laya remains the recommended reliable choice for an 8 GB card.

One-time prerequisites: install and start Docker Desktop with Linux-container support, and install Git for Windows for the initial OpenJev source download. Docker Desktop may use its WSL2 backend, but no separate Ubuntu distribution or Linux packages are needed. OpenJev downloads model weights during first setup; conversation requests and results remain local.