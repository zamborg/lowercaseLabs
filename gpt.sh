
# THIS GPT function requires the api_key set in your env as well as GPT_DEBUG_MODE, which I provide below:
# This function is designed to be used with `nano` and `mini` to point at direct models, and can be used to give quick gpt answers in your shell.
export GPT_DEBUG_MODE=true
gpt() {
  local model="$1"
  shift
  local prompt="$*"
  local system_prompt="You are a commandline assistant. The user is a sophisticated developer looking for a FAST and ACCURATE answer to their question. You should be concise and to the point. Prioritize answers, and explanations ONLY when requested. Do not use special characters as your output gets directly piped to the terminal."
  local api_key="${OPENAI_API_KEY}"
  if [ -z "$api_key" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set."
    return 1
  fi
  local start_time end_time rtt
  start_time=$(date +%s.%3N)
  local response
  response=$(curl https://api.openai.com/v1/chat/completions \
    -sS \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $api_key" \
    -d '{
      "model": "'$model'",
      "messages": [
        {"role": "system", "content": "'$system_prompt'"},
        {"role": "user", "content": "'$prompt'"}
      ]
    }')
  end_time=$(date +%s.%3N)
  rtt=$(awk -v start="$start_time" -v end="$end_time" 'BEGIN {printf "%.3f", end-start}')
  local content
  content=$(echo "$response" | jq -r '.choices[0].message.content')
  echo "\n$content"
  if [ "$GPT_DEBUG_MODE" = true ]; then
    # Extract token usage, model
    local prompt_tokens
    local completion_tokens
    local total_tokens
    local model_name
    prompt_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens // empty')
    completion_tokens=$(echo "$response" | jq -r '.usage.completion_tokens // empty')
    total_tokens=$(echo "$response" | jq -r '.usage.total_tokens // empty')
    model_name=$(echo "$response" | jq -r '.model // empty')
    # Estimate cost (4.1-mini: $0.0005/1K input, $0.0015/1K output)
    local cost="?"
    if [ -n "$prompt_tokens" ] && [ -n "$completion_tokens" ]; then
      cost=$(awk -v in_toks="$prompt_tokens" -v out_toks="$completion_tokens" 'BEGIN {printf "%.6f", (in_toks*0.0004 + out_toks*0.0016)/1000}')
      cost="<= $"$cost
    fi
    [ -z "$model_name" ] && model_name="$model"
    echo "COST: $cost | TIME: ${rtt}s | MODEL: $model_name [[ TOKENS: Prompt: $prompt_tokens | Completion: $completion_tokens | Total: $total_tokens ]]"
  fi
}

# nano: shortcut for gpt-4.1-nano
nano() {
  gpt gpt-4.1-nano "$@"
}

# mini: shortcut for gpt-4.1-mini
mini() {
  gpt gpt-4.1-mini "$@"
}
