class CostTracker:
    def __init__(self, price_per_1k_tokens=0.002):
        self.total_tokens = 0
        self.total_calls = 0
        self.price_per_1k_tokens = price_per_1k_tokens

    def estimate_tokens(self, text):
        return len(text) // 4  # rough estimate

    def track(self, prompt, response):
        prompt_tokens = self.estimate_tokens(prompt)
        response_tokens = self.estimate_tokens(response)

        total = prompt_tokens + response_tokens

        self.total_tokens += total
        self.total_calls += 1

        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total,
            "estimated_cost": (total / 1000) * self.price_per_1k_tokens
        }

    def summary(self):
        return {
            "total_api_calls": self.total_calls,
            "total_tokens_used": self.total_tokens,
            "estimated_total_cost": (self.total_tokens / 1000) * self.price_per_1k_tokens
        }
