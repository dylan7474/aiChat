# aiChat

## Overview
aiChat is a terminal-based arena where two Ollama-hosted language models take turns discussing a user-supplied topic. The
application orchestrates the conversation by prompting each model sequentially, collecting their responses via the Ollama
HTTP API, and printing the exchange with simple visual cues so you can observe how the models interact.

## Build Instructions
Before building, run the provided `./configure` script to verify that the required toolchain and libraries are available. The
script will point you toward installation commands if anything is missing.

On Debian/Ubuntu systems you can install all required build dependencies with:

```
sudo apt-get update
sudo apt-get install -y \
  build-essential make gcc pkg-config \
  libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
  libjson-c-dev
```

### Linux (GNU Make)
1. `./configure`
2. `make`
3. Run the resulting `./aichat` binary.

### Windows (MinGW / MSYS2 Make)
1. Ensure you are in an MSYS2 or MinGW shell with build tools available.
2. `./configure`
3. `make -f Makefile.win`
4. Run the generated `aichat.exe` from the same shell.

## Basic Controls
* When the program launches it prints the configured model names and prompts for an initial talking point.
* Type a topic and press <kbd>Enter</kbd> to start the exchange.
* The application alternates between the two models for the configured number of turns, displaying each response with a
  distinguishing icon and color.
* Press <kbd>Ctrl</kbd>+<kbd>C</kbd> at any time to abort the session.

## Roadmap
* Make the conversation length configurable via command-line flags.
* Allow dynamic selection of Ollama models at runtime.
* Add streaming output to display responses as they arrive from the server.
* Provide a transcript export option (text/JSON) after the session ends.

## License
aiChat is distributed under the MIT License. See [LICENSE](LICENSE) for details.
