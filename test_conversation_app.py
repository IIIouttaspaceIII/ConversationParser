import os
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from conversation_app import (
    CONTAINER_NAME,
    HUGGINGFACE_VOLUME,
    MODELS,
    ensure_images,
    ensure_server,
    normalize_shell_scripts,
    output_path_for,
    process_files,
    select_device,
    select_model,
    start_server,
)


class LauncherTests(unittest.TestCase):
    def test_normalizes_crlf_in_shell_scripts(self):
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder)
            script = checkout / "docker" / "entrypoint.sh"
            script.parent.mkdir()
            script.write_bytes(b"#!/usr/bin/env bash\r\necho ready\r\n")

            normalize_shell_scripts(checkout)

            self.assertEqual(script.read_bytes(), b"#!/usr/bin/env bash\necho ready\n")

    def test_model_selection_defaults_to_small_model(self):
        self.assertEqual(select_model(input_fn=lambda _: "").api_name, "verdict-1.4")
        self.assertEqual(select_model(input_fn=lambda _: "2").api_name, "laya-1.0")
        self.assertEqual(select_model(input_fn=lambda _: "3").api_name, "jevk5-0.2")
        self.assertEqual(select_model(input_fn=lambda _: "4").api_name, "clm-v0.1")
        self.assertEqual(select_model(input_fn=lambda _: "5").api_name, "openjev-latest")
        self.assertIsNone(select_model(input_fn=lambda _: "q"))

    def test_model_selection_lists_models_from_running_server(self):
        option = select_model(
            input_fn=lambda _: "3",
            available_models={"openjev-latest", "clm-v0.1", "jevk5-0.2"},
        )
        self.assertEqual(option.api_name, "openjev-latest")
        self.assertIsNone(option.backend)

    def test_device_selection_can_choose_cuda_or_cpu(self):
        self.assertEqual(select_device(input_fn=lambda _: "2", gpu_available=True), "cuda")
        self.assertEqual(select_device(input_fn=lambda _: "", gpu_available=True), "cpu")
        choices = iter(("2", "1"))
        self.assertEqual(select_device(input_fn=lambda _: next(choices), gpu_available=False), "cpu")
        self.assertEqual(select_device(input_fn=lambda _: "", gpu_available=True, gpu_only=True), "cuda")
        with self.assertRaisesRegex(RuntimeError, "requires an NVIDIA GPU"):
            select_device(gpu_available=False, gpu_only=True)

    @patch("conversation_app.conversation_ranker.available_models", return_value={"verdict-1.4"})
    def test_reuses_matching_local_server(self, _available_models):
        self.assertIsNone(ensure_server(MODELS[0]))

    @patch("conversation_app.image_exists", side_effect=(False, False))
    @patch("conversation_app.subprocess.run")
    def test_builds_shared_base_and_selected_backend(self, run, _image_exists):
        image = ensure_images(Path("C:/openjev"), MODELS[0])
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn("docker/Dockerfile.base", commands[0])
        self.assertIn("docker/Dockerfile.verdict", commands[1])
        self.assertIn("BASE_IMAGE=razorback16/openjev-base:cu130-torch2.13", commands[1])
        self.assertEqual(image, "conversation-block-ranker/openjev-verdict:local")

    @patch("conversation_app.image_exists", return_value=True)
    @patch("conversation_app.subprocess.run")
    def test_main_model_uses_openjev_dockerfile(self, run, _image_exists):
        ensure_images(Path("C:/openjev"), MODELS[4])
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--file") + 1], "docker/Dockerfile")
        self.assertEqual(command[command.index("--tag") + 1], "conversation-block-ranker/openjev-openjev:local")

    @patch("conversation_app.ensure_huggingface_volume")
    @patch("conversation_app.subprocess.run")
    def test_gpu_container_is_loopback_bound_and_uses_persistent_model_cache(self, run, _volume):
        run.return_value = CompletedProcess([], 0, stdout="", stderr="")
        self.assertEqual(start_server(MODELS[0], "cuda", "ranker/openjev-verdict:local"), CONTAINER_NAME)
        command = run.call_args.args[0]
        gpu_check = next(call.args[0] for call in run.call_args_list if "--entrypoint" in call.args[0])
        self.assertIn("127.0.0.1:8080:8080", command)
        self.assertIn(f"{HUGGINGFACE_VOLUME}:/root/.cache/huggingface", command)
        self.assertEqual(command[command.index("--gpus") + 1], "all")
        self.assertEqual(gpu_check[gpu_check.index("--entrypoint") + 1], "python")
        self.assertLess(gpu_check.index("--entrypoint"), gpu_check.index("ranker/openjev-verdict:local"))

    @patch("conversation_app.ensure_huggingface_volume")
    @patch("conversation_app.subprocess.run")
    def test_gpu_util_env_var_is_forwarded_when_set(self, run, _volume):
        run.return_value = CompletedProcess([], 0, stdout="", stderr="")
        with patch.dict(
            os.environ,
            {
                "OPENJEV_GPU_UTIL": "0.65",
                "OPENJEV_MAX_MODEL_LEN": "4096",
                "OPENJEV_VLLM_ARGS": "--cpu-offload-gb 4",
            },
            clear=False,
        ):
            start_server(MODELS[0], "cuda", "ranker/openjev-verdict:local")
        command = run.call_args.args[0]
        self.assertIn("OPENJEV_GPU_UTIL=0.65", command)
        self.assertIn("OPENJEV_MAX_MODEL_LEN=4096", command)
        self.assertIn("OPENJEV_VLLM_ARGS=--cpu-offload-gb 4", command)

    def test_output_path_avoids_overwriting_results(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "conversation.txt"
            source.touch()
            first = output_path_for(source)
            first.touch()
            self.assertEqual(output_path_for(source).name, "conversation_top_blocks_2.json")

    @patch("conversation_app.conversation_ranker.main", return_value=0)
    def test_file_prompt_processes_file_then_accepts_close(self, ranker_main):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "chat export.txt"
            source.write_text("conversation text", encoding="utf-8")
            answers = iter((f'"{source}"', "Does this mention vision?", "", "q"))

            process_files(MODELS[0], input_fn=lambda _: next(answers))

        arguments = ranker_main.call_args.args[0]
        self.assertIn("--model", arguments)
        self.assertIn("verdict-1.4", arguments)
        self.assertIn("Does this mention vision?", arguments)
        output = Path(arguments[arguments.index("--output") + 1])
        self.assertEqual(output.name, "chat export_top_blocks.json")


if __name__ == "__main__":
    unittest.main()