from pydantic import BaseModel

class TokenCount(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens + self.thinking_tokens

    def __str__(self):
        return f"Total Tokens: {self.total_tokens} [ Input: {self.prompt_tokens} || Completion: {self.completion_tokens} || Thinking: {self.thinking_tokens} ]"