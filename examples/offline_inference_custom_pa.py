###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from vllm import LLM, SamplingParams

import os
os.environ["VLLM_PROMPT_BS_BUCKET_MIN"] = "128"
os.environ["VLLM_PROMPT_BS_BUCKET_STEP"] = "1"
os.environ["VLLM_PROMPT_BS_BUCKET_MAX"] = "128"
os.environ["VLLM_PROMPT_SEQ_BUCKET_MIN"] = "384"
os.environ["VLLM_PROMPT_SEQ_BUCKET_MAX"] = "384"
os.environ["VLLM_DECODE_BS_BUCKET_MIN"] = "128"
os.environ["VLLM_DECODE_BS_BUCKET_MAX"] = "128"
os.environ["VLLM_DECODE_BLOCK_BUCKET_MIN"] = "256"
os.environ["VLLM_DECODE_BLOCK_BUCKET_MAX"] = "256"
os.environ["VLLM_PREFILL_USE_FUSESDAPA"] = "1"
# Sample prompts.
prompt_ids = [
    #[128000, 19930, 264, 1160, 315, 78888, 369, 52104, 8158, 369, 420, 10960, 551, 2485, 1129, 82, 2642, 20306, 21100, 4970, 31607, 14],
    #[128000, 12947, 369, 11821, 279, 3388, 7384, 369, 279, 1176, 3116, 5672, 13, 3234, 499, 617, 904, 4860, 477, 374, 1070, 4205, 775, 358, 649, 7945, 499, 449, 30],
    [128000, 96556, 0, 5810, 527, 220, 605, 10507, 315, 96894, 11380, 15174, 1473, 16, 13, 31753, 2768, 449, 13711, 25, 1115, 8446, 18065, 1701, 5370, 13711, 11, 1778, 439, 7366, 49920, 477, 8844, 8333, 1963, 11, 311, 10765, 51950, 12032, 323, 1243, 4737, 10093, 304, 1884, 12032, 3196, 389, 279, 5216, 315, 279, 9327, 627, 17, 13, 35341, 8997, 11380, 25, 1115, 8446, 18065, 4737, 10093, 14329, 311, 279, 8857, 315, 3157, 13324, 11, 389, 279, 25329, 430, 279, 13734, 374, 3629, 5076, 627, 18, 13, 41824, 12984, 25, 1115, 8446, 18065, 28865, 1990, 2204, 26593, 315, 279, 3157, 3196, 389, 872, 8844, 8333, 477, 23948, 627, 19, 13, 3749, 32505, 11380, 25, 1115, 8446, 18065, 4737, 10093, 3196, 389, 3230, 4455, 11, 1778, 439, 24608, 45976, 477, 23331, 4442, 627, 20, 13, 5513, 6108, 11380, 25, 1115, 8446, 18065, 4737, 10093, 3196, 389, 3754, 323, 1023, 17880, 2561, 2038, 627, 21, 13, 24248, 3904, 6492, 25, 1115, 8446, 18065, 4737, 10093, 3196, 389, 279, 27065, 315, 3157, 13324, 11, 439, 27000, 304, 3674, 3772, 477, 1023, 8336, 315, 828, 627, 22, 13, 27766, 6492, 25, 1115, 8446, 18065, 4737, 10093, 3196, 389, 9676, 12912, 323, 1023, 11156, 34824, 11, 1778, 439, 1862, 323, 13957, 5990, 627, 23, 13, 92539, 6492, 25, 1115, 8446, 18065, 4737, 10093, 3196, 389, 279, 16940, 6020, 323, 7100, 4787, 315, 279, 12032, 1694, 31207, 11, 1778, 439, 24608, 323, 6650, 4754, 627, 24, 13, 15996, 412, 11380, 25, 1115, 8446, 18065, 4737, 10093, 994, 279, 3430, 18808, 3485, 264, 13957, 2237, 477, 3770, 264, 1862, 2237, 11, 449, 279, 25329, 430, 279, 63788, 374, 264, 5199, 1567, 627, 605, 13, 96210, 11380, 25, 1115, 8446, 18065, 4737, 10093, 304, 12032, 449, 1579, 24151, 11, 389, 279, 25329, 430, 279, 24151, 690, 3136, 13],
    #[128000, 19930, 264, 1160, 315, 78888, 369, 52104, 8158, 369, 420, 10960, 551, 2485, 1129, 82, 2642, 20306, 21100, 4970, 31607, 14],
]

static_ids = []
input_seq_len = 127
batch_size = 128
for seq in prompt_ids:
    static_seq = (seq * (input_seq_len // len(seq) + 1))[:input_seq_len]
    static_ids.append(static_seq)

static_ids = (static_ids * (batch_size // len(static_ids) + 1))[:batch_size]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="/home/smuradi/Documents/poc/llama3/Meta-Llama-3-8B",
          trust_remote_code=True,
          dtype="bfloat16",
          block_size=128,
          max_num_seqs=128,
          num_lookahead_slots=1,
          use_v2_block_manager=True,
        #   enable_delayed_sampling=True,
          gpu_memory_utilization=0.85)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompt_token_ids=static_ids, sampling_params=sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
