# aiChat

## Overview
aiChat now runs as a lightweight local web server that lets you configure a roster of Ollama-hosted models and watch them
debate a topic inside your browser. Define friendly display names, choose which model backs each participant, pick the
number of turns, and even invite more than two companions to the discussion. The application orchestrates the entire
exchange via the Ollama HTTP API and streams the finished transcript back to the web UI.

## Build Instructions
Before building, run the provided `./configure` script to verify that the required toolchain and libraries are available. The
script will point you toward installation commands if anything is missing.

On Debian/Ubuntu systems you can install all required build dependencies with:

```
sudo apt-get update
sudo apt-get install -y \
  build-essential make gcc pkg-config \
  libcurl4-openssl-dev \
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

The server listens on port `8080` by default. If that port is already taken, aiChat automatically falls back to an
available port and prints the new URL in the terminal. Override the port explicitly by setting `AICHAT_PORT`, e.g.
`AICHAT_PORT=9000 ./aichat`. The Ollama API endpoint defaults to `http://127.0.0.1:11434/api/generate` and can be changed
by exporting `OLLAMA_URL`.

## Basic Controls
1. Run `./aichat` and wait for the console message confirming the URL (defaults to `http://127.0.0.1:8080`).
2. Open the address in a web browser.
3. Enter a topic, choose the number of turns (each participant speaks once per turn), and configure as many participants as
   you want—each with a friendly name and Ollama model tag.
4. Press **Start conversation** to watch the transcript appear in the browser.
5. Remove or add participants at any time using the on-page controls.

If you prefer to stop the server, press <kbd>Ctrl</kbd>+<kbd>C</kbd> in the terminal where the binary is running.

## Roadmap
* Add streaming output to display responses as they arrive from the server.
* Provide a transcript export option (text/JSON) after the session ends.
* Allow saving and reusing favourite participant rosters.

## License
aiChat is distributed under the MIT License. See [LICENSE](LICENSE) for details.
