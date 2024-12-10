###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
from vllm import LLM, SamplingParams
import os
os.environ["VLLM_PROMPT_BS_BUCKET_MIN"] = "1"
os.environ["VLLM_PROMPT_BS_BUCKET_STEP"] = "1"
os.environ["VLLM_PROMPT_BS_BUCKET_MAX"] = "4"
os.environ["VLLM_PROMPT_SEQ_BUCKET_MIN"] = "2048"
os.environ["VLLM_PROMPT_SEQ_BUCKET_MAX"] = "2048"
os.environ["VLLM_DECODE_BS_BUCKET_MIN"] = "128"
os.environ["VLLM_DECODE_BS_BUCKET_MAX"] = "128"
os.environ["VLLM_DECODE_BLOCK_BUCKET_MIN"] = "1024"
os.environ["VLLM_DECODE_BLOCK_BUCKET_STEP"] = "1024"
os.environ["VLLM_DECODE_BLOCK_BUCKET_MAX"] = "4096"
os.environ["VLLM_PROMPT_USE_FUSEDSDPA"] = "1"
os.environ["VLLM_CONTIGUOUS_PA"] = "false"
os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["VLLM_CUSTOM_PA"] = "true"
os.environ["VLLM_CUSTOM_PA_STORE_KEY"] = "true"
# Sample prompts.
# Prompt: None, Generated text: ' art,\nAnd, constant stars, in them I read such art,\nAnd,'
# Prompt: None, Generated text: ',\nAnd all in war with Time for love of you,\nAnd all in war'
prompt_ids = [
    [128000, 128000, 128006, 882, 128007, 271, 38053, 439, 1690, 5238, 439, 499, 649, 505, 1521, 33894, 5238, 512, 31193, 20028, 267, 20566, 584, 12876, 5376, 345, 4897, 28592, 13444, 596, 16392, 2643, 2646, 2815, 345, 4071, 439, 279, 436, 13154, 1288, 555, 892, 31952, 521, 345, 16366, 28682, 51543, 2643, 11984, 813, 5044, 512, 4071, 34223, 11, 51068, 311, 270, 483, 1866, 10107, 6548, 345, 30016, 596, 83, 26236, 3177, 596, 83, 35678, 449, 659, 18451, 77057, 10633, 345, 43346, 264, 79054, 1405, 37492, 15812, 345, 2645, 12732, 1855, 304, 1855, 555, 27848, 22106, 345, 4438, 3117, 358, 311, 321, 11, 2103, 43726, 1022, 505, 40344, 627, 23956, 358, 502, 2343, 439, 422, 539, 7318, 1603, 627, 64595, 387, 26236, 3021, 323, 26236, 3021, 596, 1005, 872, 32726, 627, 69071, 11, 422, 34223, 75596, 11, 34223, 1989, 28530, 315, 1690, 345, 3112, 41398, 82673, 315, 459, 47691, 5609, 512, 51, 484, 22037, 11, 439, 1364, 79703, 40344, 11, 11299, 264, 1773, 11780, 345, 6, 56568, 856, 4333, 596, 51108, 15042, 449, 420, 7982, 4325, 345, 644, 40344, 26236, 7474, 11, 39357, 34223, 387, 1612, 484, 4265, 512, 12487, 1427, 358, 4648, 856, 2919, 1288, 1367, 6629, 627, 69071, 11, 422, 34223, 75596, 11, 34223, 1989, 28530, 315, 1690, 345, 33763, 34223, 449, 586, 45972, 34662, 757, 345, 6, 56568, 856, 4333, 596, 51108, 15042, 449, 420, 7982, 4325, 345, 4897, 4245, 315, 1690, 1457, 374, 270, 483, 7636, 512, 1090, 408, 264, 14017, 30543, 90413, 311, 40344, 345, 47, 9773, 26236, 12737, 311, 856, 14254, 1752, 1684, 345, 3112, 555, 5369, 757, 315, 40344, 24164, 345, 3112, 781, 589, 15031, 649, 7197, 6439, 304, 10437, 478, 37808, 627, 1016, 283, 430, 1989, 1457, 279, 1917, 596, 7878, 79760, 198, 46, 6, 1095, 757, 11, 837, 304, 3021, 11, 719, 9615, 3350, 345, 12487, 1268, 11, 994, 7138, 6880, 40344, 311, 387, 8208, 345, 18293, 682, 856, 6981, 315, 26236, 5922, 323, 8206, 627, 1271, 279, 7029, 1917, 323, 682, 1077, 59617, 76657, 280, 1671, 974, 8333, 596, 37492, 7621, 729, 813, 1866, 4851, 627, 1844, 7349, 4265, 389, 1891, 478, 11, 7389, 34223, 636, 264, 4538, 627, 18293, 682, 856, 6981, 315, 26236, 5922, 323, 8206, 627, 1383, 43146, 7168, 430, 358, 304, 23070, 1505, 512, 8238, 40344, 2500, 659, 11, 369, 3021, 315, 757, 345, 4516, 1317, 6439, 420, 323, 420, 6835, 2324, 311, 40344, 627, 10086, 912, 26314, 25, 1243, 499, 1051, 198, 4516, 1317, 439, 3026, 649, 37397, 477, 6548, 649, 1518, 345, 1016, 88, 12822, 596, 12691, 326, 6633, 11, 779, 342, 28109, 389, 1457, 345, 2520, 40344, 323, 369, 7182, 912, 11594, 1505, 627, 4071, 1938, 656, 339, 7446, 4128, 856, 25551, 1849, 5129, 198, 4897, 4245, 315, 1690, 1457, 374, 270, 483, 7636, 512, 10445, 1550, 267, 34223, 11471, 1778, 264, 387, 64, 1088, 788, 1938, 345, 3112, 10051, 872, 34300, 1614, 704, 315, 5044, 280, 3968, 1683, 279, 16700, 2442, 3806, 449, 46384, 8071, 345, 3112, 4400, 364, 60246, 267, 4212, 596, 1156, 96960, 649, 1304, 23682, 198, 2822, 5129, 18821, 1109, 499, 6261, 1618, 3974, 512, 2244, 311, 26236, 726, 520, 3325, 3169, 70395, 12391, 512, 4897, 358, 304, 26236, 37492, 1097, 8657, 7725, 198, 1016, 283, 315, 26236, 726, 26236, 10437, 659, 49415, 93928, 627, 791, 12703, 2759, 315, 2291, 1481, 6868, 52530, 4647, 276, 345, 1016, 88, 842, 374, 8206, 596, 323, 13444, 596, 59714, 323, 2457, 627, 8238, 4208, 5304, 420, 36277, 43049, 519, 11, 4212, 5380, 1271, 990, 856, 4059, 11, 994, 2547, 596, 990, 596, 27489, 512, 40, 3371, 279, 1938, 11, 311, 4587, 1124, 34223, 1989, 10107, 198, 791, 17260, 369, 872, 1742, 358, 3358, 1373, 11, 813, 369, 813, 3021, 24314, 1016, 88, 13444, 596, 1376, 304, 2007, 315, 856, 4851, 280, 4071, 539, 311, 3371, 315, 1695, 477, 14289, 15369, 345, 3112, 1475, 6762, 449, 813, 6762, 656, 339, 48306, 2648, 198, 1383, 29590, 12502, 11, 656, 85086, 270, 483, 2487, 345, 4599, 36498, 86082, 4985, 293, 2423, 7404, 26236, 60375, 345, 27588, 939, 4668, 1752, 323, 47101, 11, 95088, 398, 83217, 512, 50, 826, 420, 311, 40344, 25, 364, 339, 283, 3254, 75596, 12391, 7000, 24314, 39, 561, 34223, 11, 279, 7491, 1474, 380, 676, 315, 856, 11939, 280, 1383, 29590, 12502, 11, 656, 85086, 270, 483, 2487, 345, 10086, 912, 26314, 25, 1243, 499, 1051, 198, 3112, 304, 5694, 872, 22519, 15812, 28016, 345, 16366, 28682, 51543, 2643, 11984, 813, 5044, 512, 11787, 11276, 311, 856, 17659, 11, 1405, 43847, 279, 7160, 198, 11787, 11276, 311, 856, 17659, 11, 1405, 43847, 279, 7160, 198, 3112, 89635, 555, 33415, 3131, 810, 312, 1355, 39378, 198, 2170, 8206, 323, 13444, 4985, 3871, 41972, 345, 10149, 1514, 279, 43049, 1821, 311, 279, 1633, 1890, 198, 3112, 555, 264, 961, 315, 682, 26236, 27025, 3974, 627, 3112, 10051, 872, 34300, 1614, 704, 315, 5044, 280, 2520, 912, 893, 1664, 315, 1778, 264, 4371, 588, 649, 6604, 198, 4897, 358, 304, 26236, 37492, 1097, 8657, 7725, 198, 1383, 6140, 477, 7138, 596, 10223, 3388, 653, 10893, 76, 4265, 280, 4897, 88084, 279, 27653, 323, 272, 1439, 539, 279, 64935, 512, 1079, 336, 9894, 3831, 12822, 304, 813, 6278, 4325, 345, 644, 40344, 26236, 7474, 11, 39357, 34223, 387, 1612, 484, 4265, 512, 4897, 23070, 596, 3805, 304, 420, 6908, 89781, 554, 305, 12116, 627, 11356, 1304, 2873, 12743, 11, 304, 19762, 4339, 311, 1501, 433, 345, 3112, 1518, 26236, 6680, 8369, 994, 34223, 2733, 596, 83, 433, 9439, 627, 3112, 11, 6926, 9958, 11, 304, 1124, 358, 1373, 1778],
    [128000, 128000, 128006, 882, 128007, 271, 38053, 439, 1690, 5238, 439, 499, 649, 505, 1521, 33894, 5238, 512, 31193, 20028, 267, 20566, 584, 12876, 5376, 345, 4897, 28592, 13444, 596, 16392, 2643, 2646, 2815, 345, 4071, 439, 279, 436, 13154, 1288, 555, 892, 31952, 521, 345, 16366, 28682, 51543, 2643, 11984, 813, 5044, 512, 4071, 34223, 11, 51068, 311, 270, 483, 1866, 10107, 6548, 345, 30016, 596, 83, 26236, 3177, 596, 83, 35678, 449, 659, 18451, 77057, 10633, 345, 43346, 264, 79054, 1405, 37492, 15812, 345, 8100, 49267, 40344, 369, 1077, 26418, 11, 323, 8967, 28592, 198, 791, 17104, 36496, 1405, 1475, 8071, 656, 339, 44935, 345, 5159, 9168, 4985, 539, 51041, 757, 358, 1097, 2362, 345, 4897, 358, 304, 26236, 37492, 1097, 8657, 7725, 198, 791, 47753, 596, 58596, 79018, 719, 7621, 16337, 198, 2409, 32931, 2349, 11, 439, 374, 905, 3278, 596, 11401, 280, 27831, 3686, 11, 23070, 8964, 11, 433, 374, 719, 439, 264, 44180, 198, 3112, 682, 304, 4208, 449, 4212, 369, 3021, 315, 499, 345, 46, 11, 4048, 311, 1373, 1148, 21737, 3021, 52677, 2155, 512, 791, 293, 632, 34172, 4143, 434, 2728, 40344, 311, 3041, 5380, 3957, 505, 279, 2363, 315, 34662, 24788, 291, 5115, 345, 23956, 358, 502, 2343, 439, 422, 539, 7318, 1603, 627, 10149, 1514, 279, 43049, 1821, 311, 279, 1633, 1890, 198, 2244, 2019, 449, 84293, 422, 433, 4985, 733, 1664, 345, 791, 832, 555, 311, 321, 11, 279, 1023, 311, 29011, 198, 15546, 682, 304, 832, 11, 832, 54799, 5296, 656, 7936, 512, 4071, 1405, 1348, 656, 539, 499, 264, 2643, 1291, 1648, 198, 3112, 682, 304, 4208, 449, 4212, 369, 3021, 315, 499, 345, 1271, 11550, 14523, 11, 539, 311, 1501, 856, 38467, 512, 4897, 4245, 315, 1690, 1457, 374, 270, 483, 7636, 512, 23956, 293, 632, 34172, 8352, 34223, 1288, 267, 304, 53523, 87785, 512, 6153, 264, 16579, 46146, 3131, 47499, 4265, 345, 1271, 1095, 2385, 30614, 297, 6, 531, 731, 757, 304, 856, 1648, 345, 2746, 34223, 1436, 267, 4320, 364, 2028, 6762, 1716, 315, 10705, 198, 3112, 682, 304, 4208, 449, 4212, 369, 3021, 315, 499, 345, 7184, 374, 279, 892, 430, 3663, 1288, 1376, 2500, 280, 1016, 88, 842, 374, 8206, 596, 323, 13444, 596, 59714, 323, 2457, 627, 1090, 408, 264, 14017, 30543, 90413, 311, 40344, 345, 4897, 420, 6908, 6566, 3118, 774, 308, 2509, 719, 5039, 198, 4897, 34223, 912, 1376, 315, 40344, 34143, 2163, 4920, 345, 68397, 1752, 603, 7889, 11, 3249, 49415, 34223, 1005, 198, 10596, 11, 1148, 374, 1888, 11, 430, 1888, 358, 6562, 304, 40344, 512, 2822, 810, 387, 342, 83712, 520, 430, 902, 34223, 34143, 2884, 512, 1687, 661, 449, 311, 321, 11, 358, 90539, 757, 311, 856, 4950, 345, 4516, 1317, 6439, 420, 323, 420, 6835, 2324, 311, 40344, 627, 9673, 8009, 47101, 5238, 315, 26236, 43720, 31657, 345, 3112, 505, 279, 369, 75, 1540, 1917, 813, 2145, 425, 10477, 345, 14704, 3972, 539, 389, 26236, 4851, 994, 10705, 374, 70286, 280, 4071, 358, 74731, 40344, 832, 1455, 568, 61798, 9977, 512, 89553, 459, 682, 5773, 1113, 21648, 323, 82642, 1752, 29488, 627, 9241, 358, 1253, 539, 4148, 6463, 387, 7108, 627, 89553, 459, 682, 5773, 1113, 21648, 323, 82642, 1752, 29488, 627, 3112, 7474, 596, 6307, 682, 342, 2668, 291, 709, 304, 1364, 4798, 198, 3112, 584, 752, 8136, 2548, 3021, 596, 1317, 2533, 18974, 4265, 289, 4748, 345, 2127, 263, 11810, 279, 3122, 478, 30614, 311, 12141, 198, 27831, 304, 1057, 6439, 264, 4941, 481, 34781, 345, 10596, 11, 1148, 459, 653, 339, 42480, 304, 279, 1917, 656, 339, 8493, 198, 2746, 279, 837, 3613, 541, 315, 1664, 2442, 49983, 10578, 345, 79519, 596, 387, 724, 6835, 4400, 719, 656, 339, 39580, 345, 4897, 1097, 4316, 1138, 4265, 279, 8935, 315, 2800, 5380, 42821, 985, 26236, 40444, 810, 1109, 26236, 40444, 527, 280, 10267, 757, 48466, 430, 584, 1403, 2011, 387, 4483, 467, 345, 644, 832, 315, 270, 483, 11, 505, 430, 902, 34223, 11776, 478, 280, 6153, 264, 16579, 46146, 3131, 47499, 4265, 345, 4071, 1938, 555, 3814, 11, 323, 3814, 555, 1938, 11, 90838, 4265, 5380, 4897, 1243, 358, 88106, 311, 2349, 856, 1614, 449, 45619, 627, 10267, 757, 48466, 430, 584, 1403, 2011, 387, 4483, 467, 345, 3112, 3515, 26438, 4265, 279, 32366, 5352, 67698, 24898, 345, 4599, 358, 45493, 430, 3026, 439, 11012, 5376, 345, 56392, 1203, 279, 17104, 5936, 315, 1077, 10461, 512, 1271, 6865, 449, 6548, 17623, 311, 3021, 596, 7060, 38467, 627, 9241, 263, 279, 9958, 304, 6367, 10383, 4068, 280, 2746, 34223, 1436, 267, 4320, 364, 2028, 6762, 1716, 315, 10705, 198, 15546, 682, 304, 832, 11, 832, 54799, 5296, 656, 7936, 512, 791, 6548, 11, 364, 1348, 294, 1088, 788, 11, 1457, 16489, 527, 198, 39, 6714, 26236, 86166, 304, 872, 78792, 16603, 5380, 3112, 682, 304, 4208, 449, 4212, 369, 3021, 315, 499, 345, 2746, 34223, 18167, 856, 1664, 6951, 291, 1938, 345, 56392, 1203, 279, 17104, 5936, 315, 1077, 10461, 512, 39, 589, 25237, 10597, 3021, 106470, 44886, 505, 10705, 8071, 198, 1016, 88, 38559, 52677, 856, 14523, 16917, 53203, 345, 3112, 1193, 65206, 311, 279, 342, 8039, 88, 10683, 345, 35, 20850, 779, 2294, 11, 902, 38467, 779, 8009, 439, 10705, 198, 9241, 41421, 52221, 4212, 4316, 266, 774, 449, 98386, 345, 40, 1253, 539, 3596, 6518, 25670, 40344, 345, 2520, 34223, 1989, 779, 15575, 4265, 449, 85229, 12491, 198, 27831, 3686, 11, 23070, 8964, 11, 433, 374, 719, 439, 264, 44180, 198, 4599, 304, 35825, 5238, 311, 892, 34223, 3139, 478, 25, 128009, 128006],
]
prompt_ids = (prompt_ids * (128 // len(prompt_ids) + 1))[:128]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)
# Create an LLM.
llm = LLM(model="/mnt/weka/data/pytorch/llama3/Meta-Llama-3-8B/",
          trust_remote_code=True,
          dtype="bfloat16",
          block_size=128,
          max_num_seqs=128,
          num_lookahead_slots=1,
          use_v2_block_manager=True,
          gpu_memory_utilization=0.85)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
