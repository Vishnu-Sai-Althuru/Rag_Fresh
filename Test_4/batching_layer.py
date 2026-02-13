def batch_prompts(prompts):
    print(f"Processing batch of {len(prompts)} prompts")
    from llm_client import LLMClient
    client = LLMClient()

    responses = []
    for prompt in prompts:
        responses.append(client.generate(prompt))

    return responses