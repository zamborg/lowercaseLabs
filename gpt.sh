gpt() {
  local data='{
      "model": "gpt-3.5-turbo",
      "messages": [
        {"role": "system", "content": $prompt},
        {"role": "user", "content": $query}
      ]
    }'
  local prompt="You are a helpful assistant." #change this if you'd like
  local query="$*"
  data=$(jq -n --arg query "$query" --arg prompt "$prompt" $data)

  local response=$(curl "https://api.openai.com/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \ # your api key
    -d "$data"
  )

  local linebreak="================="

  response=$(echo "$response" | jq -r '.choices[0].message.content')

  echo "\n$linebreak\n$response\n$linebreak"

}
