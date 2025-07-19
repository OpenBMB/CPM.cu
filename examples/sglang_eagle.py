import sglang as sgl
from transformers import AutoTokenizer

def main():
    # Sample prompts.
    prompts = [
        "who are you?",
    ]

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompts}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

    # Create a sampling params object.
    sampling_params = {"temperature": 0, "max_new_tokens": 30}

    # Create an LLM.
    llm = sgl.Engine(
        model_path="unsloth/Meta-Llama-3.1-8B-Instruct",
        speculative_algorithm="EAGLE",
        speculative_draft_model_path="jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B",
        speculative_num_steps=5,
        speculative_eagle_topk=8,
        speculative_num_draft_tokens=32,
        dtype="float16",
        mem_fraction_static=0.7,
        cuda_graph_max_bs=8,
    )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()
