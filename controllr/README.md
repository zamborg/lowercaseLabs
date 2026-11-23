# Controllr

Controllr gives your iPhone a voice assistant that can follow a conversation and act on it. Ask it to read a recent text, set a timer, or turn a copied plan into a note. When it needs a decision, it can ask aloud or show you a list to tap.

## Install

1. Open the **Controllr TestFlight invitation** from the person who shared this kit. Install TestFlight if prompted, then install Controllr.
2. Unzip the kit. Open the files in **Action Shortcuts** and tap **Add Shortcut** for each one. Then add **Controllr Voice 2.1**. You can also import them in Shortcuts on a Mac and let iCloud sync them to your iPhone. Keep the generated names so the voice shortcut can find its handlers.
3. Open **Controllr → Setup**. Choose OpenAI, Anthropic, or OpenRouter, enter your own API key, and select a model. Under **Voice shortcut**, select **Controllr Voice 2.1**.
4. Run **Controllr Voice 2.1** in Shortcuts and grant speech access when prompted. You can also launch it from Controllr, the Action Button, or Back Tap.

## Try it

- “Give me a tappable choice between a stretch and a short walk.”
- Copy a harmless sentence, then say “What did I just copy?”
- Say “Set a timer for 83 seconds.” Enable **Timers** in Controllr Setup for arbitrary durations on iOS 26 or later.

Use **Threads** to review or resume a conversation. In **Modules**, you can turn actions off and decide which changes ask first. Sending texts, calling contacts, editing events, and ride handoffs ask first by default. Reading Messages requires iOS 27.

Controllr stores your API key in the iPhone Keychain. Your chosen model provider receives conversation context and any lookup results you ask Controllr to use. Your provider's API terms and charges apply.
