from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import conversation_ranker


OPENJEV_REPOSITORY = "https://github.com/razorback16/openjev.git"
OPENJEV_URL = "http://127.0.0.1:8080"
OPENJEV_BASE_IMAGE = "razorback16/openjev-base:cu130-torch2.13"
OPENJEV_IMAGE_PREFIX = "conversation-block-ranker/openjev"
CONTAINER_NAME = "conversation-block-ranker-openjev"
HUGGINGFACE_VOLUME = "conversation-block-ranker-huggingface"
STARTUP_TIMEOUT_SECONDS = 1800


@dataclass(frozen=True)
class ModelOption:
    number: str
    label: str
    api_name: str
    backend: str | None
    requires_gpu: bool = False


MODELS = (
    ModelOption("1", "Verdict 151M; 512 tokens; 16 GB RAM / 2 GB VRAM", "verdict-1.4", "verdict"),
    ModelOption("2", "Laya 421M; 1,024 tokens; 16 GB RAM / 4 GB VRAM; recommended for 8 GB GPUs", "laya-1.0", "laya"),
    ModelOption("3", "JevK5 4B; 16K tokens; 32 GB RAM / 12 GB VRAM", "jevk5-0.2", "jevk5", True),
    ModelOption("4", "CLM 8B; 2K tokens; 32 GB RAM / 12 GB VRAM", "clm-v0.1", "clm", True),
    ModelOption("5", "DiffusionGemma 26B; 64K tokens; 64 GB RAM / 24 GB VRAM minimum", "openjev-latest", "openjev", True),
)


def select_model(
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    available_models: set[str] | None = None,
) -> ModelOption | None:
    if available_models:
        known_models = {option.api_name: option for option in MODELS}
        options = tuple(
            ModelOption(
                str(index),
                f"{known_models[name].label} (already running locally)" if name in known_models
                else f"{name} (already running locally)",
                name,
                None,
            )
            for index, name in enumerate(sorted(available_models), start=1)
        )
        print_fn("Choose a model served by the local OpenJev server:")
    else:
        options = MODELS
        print_fn("Choose a model to run locally:")
    for option in options:
        print_fn(f"  {option.number}. {option.label}")
    while True:
        choice = input_fn("Model number, or Q to close (default 1): ").strip().lower()
        if choice in {"q", "quit", "close"}:
            return None
        choice = choice or "1"
        for option in options:
            if choice == option.number:
                return option
        print_fn("Choose one of the listed numbers, or Q.")


def select_device(
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    gpu_available: bool | None = None,
    gpu_only: bool = False,
) -> str:
    if gpu_available is None:
        gpu_available = shutil.which("nvidia-smi") is not None
    if gpu_only and not gpu_available:
        raise RuntimeError("This model requires an NVIDIA GPU. Check the NVIDIA driver and nvidia-smi, then rerun the program.")
    if gpu_only:
        print_fn("This model runs on an NVIDIA GPU (CUDA).")
        while True:
            choice = input_fn("Press Enter to continue, or Q to close: ").strip().lower()
            if choice in {"", "1"}:
                return "cuda"
            if choice in {"q", "quit", "close"}:
                return ""
            print_fn("Press Enter to continue, or Q to close.")
    print_fn("Choose where the selected model should run:")
    print_fn("  1. CPU (default)")
    if gpu_available:
        print_fn("  2. NVIDIA GPU (CUDA)")
    while True:
        choice = input_fn("Device [1, 2, or Q to close] (default 1): ").strip().lower()
        if choice in {"q", "quit", "close"}:
            return ""
        if choice in {"", "1"}:
            return "cpu"
        if choice == "2" and gpu_available:
            return "cuda"
        print_fn("Choose a listed device or Q.")


def require_docker() -> None:
    if not shutil_which("docker"):
        raise RuntimeError("Docker Desktop is required. Install it, start it, and rerun this program.")
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Docker is not responding. Start Docker Desktop, then rerun this program.") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Docker is not ready. Start Docker Desktop and check its status. {detail}".strip())


def checkout_path() -> Path:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return local_app_data / "ConversationBlockRanker" / "openjev"


def normalize_shell_scripts(checkout: Path) -> None:
    for script in checkout.rglob("*.sh"):
        contents = script.read_bytes()
        normalized = contents.replace(b"\r\n", b"\n")
        if normalized != contents:
            script.write_bytes(normalized)


def ensure_checkout(git_runner: Callable = subprocess.run) -> Path:
    checkout = checkout_path()
    if (checkout / ".git").is_dir():
        normalize_shell_scripts(checkout)
        return checkout
    if checkout.exists():
        raise RuntimeError(f"Cannot download OpenJev because this folder already exists: {checkout}")
    if not shutil_which("git"):
        raise RuntimeError("Git is required to download OpenJev. Install Git for Windows, then rerun this program.")

    checkout.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OpenJev into {checkout} ...")
    git_runner(
        ["git", "-c", "core.autocrlf=false", "clone", "--depth", "1", OPENJEV_REPOSITORY, str(checkout)],
        check=True,
    )
    normalize_shell_scripts(checkout)
    return checkout


def shutil_which(program: str) -> str | None:
    from shutil import which

    return which(program)


def image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True, check=False
    )
    return result.returncode == 0


def ensure_images(checkout: Path, option: ModelOption) -> str:
    if option.backend is None:
        raise RuntimeError(f"Cannot install {option.api_name}; it must already be served locally.")
    base_image = OPENJEV_BASE_IMAGE
    if not image_exists(base_image):
        print("Building OpenJev's shared Docker base image. This first build may take a while and use several GB of disk space...")
        subprocess.run(
            ["docker", "build", "--tag", base_image, "--file", "docker/Dockerfile.base", "."],
            cwd=checkout,
            check=True,
        )

    image = f"{OPENJEV_IMAGE_PREFIX}-{option.backend}:local"
    print(f"Building or updating the OpenJev {option.backend} Docker image (cached layers are reused)...")
    dockerfile = "docker/Dockerfile" if option.backend == "openjev" else f"docker/Dockerfile.{option.backend}"
    subprocess.run(
        [
            "docker", "build", "--tag", image,
            "--build-arg", f"BASE_IMAGE={base_image}",
            "--file", dockerfile, ".",
        ],
        cwd=checkout,
        check=True,
    )
    return image


def ensure_huggingface_volume() -> None:
    result = subprocess.run(
        ["docker", "volume", "inspect", HUGGINGFACE_VOLUME], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        subprocess.run(["docker", "volume", "create", HUGGINGFACE_VOLUME], check=True, capture_output=True, text=True)


def start_server(option: ModelOption, device: str, image: str) -> str:
    if option.backend is None:
        raise RuntimeError(f"Cannot start {option.api_name}; it must already be served locally.")
    ensure_huggingface_volume()
    subprocess.run(["docker", "rm", "--force", CONTAINER_NAME], check=False, capture_output=True, text=True)
    command = [
        "docker", "run", "--detach", "--name", CONTAINER_NAME,
        "--publish", "127.0.0.1:8080:8080",
        "--volume", f"{HUGGINGFACE_VOLUME}:/root/.cache/huggingface",
        "--env", f"OPENJEV_DEVICE={device}",
    ]
    api_key = os.environ.get("OPENJEV_API_KEY")
    if api_key:
        command.extend(("--env", f"OPENJEV_API_KEY={api_key}"))
    gpu_util = os.environ.get("OPENJEV_GPU_UTIL")
    if gpu_util:
        command.extend(("--env", f"OPENJEV_GPU_UTIL={gpu_util}"))
    max_model_len = os.environ.get("OPENJEV_MAX_MODEL_LEN")
    if max_model_len:
        command.extend(("--env", f"OPENJEV_MAX_MODEL_LEN={max_model_len}"))
    vllm_args = os.environ.get("OPENJEV_VLLM_ARGS")
    if vllm_args:
        command.extend(("--env", f"OPENJEV_VLLM_ARGS={vllm_args}"))
    if device == "cuda":
        gpu_check = subprocess.run(
            [
                "docker", "run", "--rm", "--gpus", "all", "--entrypoint", "python", image,
                "-c", "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable in Docker'",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if gpu_check.returncode != 0:
            detail = (gpu_check.stderr or gpu_check.stdout or "").strip()
            if "bash\r" in detail or "bash\r" in repr(detail):
                raise RuntimeError(
                    "The selected Docker image has a CRLF-formatted Linux entrypoint, so the CUDA check never ran. "
                    "The launcher rebuilds the selected backend from normalized sources; rerun it to refresh the image. "
                    f"{detail}".strip()
                )
            raise RuntimeError(
                "The CUDA check failed inside the Docker image. Check Docker Desktop GPU support and the Windows NVIDIA driver. "
                f"{detail}".strip()
            )
        command.extend(("--gpus", "all"))
    command.append(image)
    print(f"Starting OpenJev on {device.upper()}. The first run may download model weights; this can take several minutes.")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"Could not start the OpenJev Docker container. {detail}".strip()) from error
    return CONTAINER_NAME


def container_is_running(container_name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format={{.State.Running}}", container_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def stop_server(container_name: str) -> None:
    subprocess.run(["docker", "stop", "--time", "10", container_name], check=False, capture_output=True, text=True)
    subprocess.run(["docker", "rm", "--force", container_name], check=False, capture_output=True, text=True)


def wait_for_server(
    option: ModelOption,
    container_name: str,
    timeout: int = STARTUP_TIMEOUT_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> None:
    deadline = clock_fn() + timeout
    next_notice = clock_fn() + 30
    api_key = os.environ.get("OPENJEV_API_KEY")
    last_error = "OpenJev is still starting"
    while clock_fn() < deadline:
        if not container_is_running(container_name):
            logs = subprocess.run(
                ["docker", "logs", "--tail", "30", container_name],
                capture_output=True,
                text=True,
                check=False,
            )
            detail = (logs.stderr or logs.stdout or "").strip()
            raise RuntimeError(f"The OpenJev Docker container exited before it became ready. {detail}".strip())
        try:
            models = conversation_ranker.available_models(OPENJEV_URL, api_key)
            if option.api_name in models:
                print(f"OpenJev is ready with {option.api_name}.")
                return
            last_error = f"The running server does not list {option.api_name}"
        except RuntimeError as error:
            last_error = str(error)

        if clock_fn() >= next_notice:
            print("Still waiting for the local model to load...")
            next_notice = clock_fn() + 30
        sleep_fn(2)
    raise RuntimeError(f"Timed out waiting for OpenJev: {last_error}")


def ensure_server(option: ModelOption, device: str = "cpu") -> str | None:
    api_key = os.environ.get("OPENJEV_API_KEY")
    try:
        running_models = conversation_ranker.available_models(OPENJEV_URL, api_key)
    except RuntimeError:
        running_models = None

    if running_models is not None:
        if option.api_name in running_models:
            print(f"Using the already-running local OpenJev server ({option.api_name}).")
            return None
        names = ", ".join(sorted(running_models)) or "no models"
        raise RuntimeError(f"A local OpenJev server is already running with {names}. Close it, then restart this program.")

    if option.backend is None:
        raise RuntimeError(f"{option.api_name} is not running locally and cannot be installed by this launcher.")
    if option.requires_gpu and device != "cuda":
        raise RuntimeError(f"{option.api_name} requires an NVIDIA GPU (CUDA).")
    require_docker()
    checkout = ensure_checkout()
    image = ensure_images(checkout, option)
    container_name = start_server(option, device, image)
    try:
        wait_for_server(option, container_name)
    except (OSError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt):
        stop_server(container_name)
        raise
    return container_name


def output_path_for(source: Path) -> Path:
    base = source.with_name(f"{source.stem}_top_blocks.json")
    if not base.exists():
        return base
    suffix = 2
    while True:
        candidate = source.with_name(f"{source.stem}_top_blocks_{suffix}.json")
        if not candidate.exists():
            return candidate
        suffix += 1


def prompt_questions(input_fn: Callable[[str], str] = input) -> list[str]:
    print("Enter a question for each block. Submit a blank line when finished.")
    questions = []
    while True:
        question = input_fn(f"Question {len(questions) + 1} (blank to finish): ").strip()
        if not question:
            return questions
        questions.append(question)


def process_files(option: ModelOption, input_fn: Callable[[str], str] = input) -> None:
    while True:
        raw_path = input_fn("Text file path (or Q to close): ").strip().strip('"')
        if raw_path.lower() in {"q", "quit", "close"}:
            return
        if not raw_path:
            continue

        source = Path(raw_path).expanduser()
        if not source.is_file():
            print(f"File not found: {source}")
            continue

        questions = prompt_questions(input_fn)
        if not questions:
            print("No questions entered; returning to the file prompt.")
            continue

        output = output_path_for(source)
        arguments = [str(source), "--model", option.api_name, "--output", str(output)]
        for question in questions:
            arguments.extend(("--question", question))
        if conversation_ranker.main(arguments) == 0:
            print(f"Results saved to {output}")


def main() -> int:
    try:
        running_models = conversation_ranker.available_models(OPENJEV_URL, os.environ.get("OPENJEV_API_KEY"))
    except RuntimeError:
        running_models = None
    option = select_model(available_models=running_models)
    if option is None:
        return 0

    device = select_device(gpu_only=option.requires_gpu) if option.backend is not None else None
    if device == "":
        return 0
    container_name = None
    try:
        container_name = ensure_server(option, device or "cpu")
        process_files(option)
        return 0
    except KeyboardInterrupt:
        print("Closing.")
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if container_name is not None:
            print("Stopping the OpenJev server started by this program...")
            stop_server(container_name)


if __name__ == "__main__":
    raise SystemExit(main())