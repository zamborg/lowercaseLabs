Here’s the “ephemeral history‐file” pattern you can drop into your shell rc (bash/zsh/fish).  It:

• Creates a temp file when your shell starts
• Deletes it when your shell exits
• Wraps the qork invocation so that every time you run qork some question:
– it appends “some question” to the temp file
– it calls the real qork passing that file as context

––––––––––––––––––––––––––––––––––––––––––––

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                              bash / zsh example                                                                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

In your ~/.bashrc or ~/.zshrc add:


 # 1) make a per‐shell temp file
 export QORK_HISTORY=$(mktemp /tmp/qork.XXXXXX)

 # 2) ensure it’s removed when your shell dies
 trap 'rm -f "$QORK_HISTORY"' EXIT

 # 3) wrap the qork command
 qork() {
   # append your query
   echo "$*" >> "$QORK_HISTORY"

   # call the real qork, passing the history
   # (adjust --history-file or --context flags to whatever qork expects)
   command qork --history-file "$QORK_HISTORY" "$@"
 }


Now every qork what’s up? will get “what’s up?” appended to /tmp/qork.xxxxxx, and that file is automatically passed back into qork on each run.  When you close the shell, the file vanishes.

––––––––––––––––––––––––––––––––––––––––––––