# Sidekick

**A real-time meeting co-pilot for VS Code.** Sidekick listens to your call,
transcribes it locally, and quietly surfaces researched answers, action items,
and questions-to-ask in a live feed — so you can stay present in the conversation.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform: Windows](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![VS Code](https://img.shields.io/badge/VS%20Code-Copilot%20Chat-007ACC)

- **Local-first speech-to-text** (faster-whisper on CPU) — no audio leaves the device.
- **Grounded research** via the Microsoft Learn API + optional keyless web search.
- **Findings feed** in a VS Code side panel — numbered, confidence-tagged, drill-down.
- Runs as an **MCP server** for GitHub Copilot Chat, plus a lightweight notification extension.

## Install

One line in PowerShell (Windows 10/11):

```powershell
irm https://raw.githubusercontent.com/Kenniola/sidekick-copilot/main/sidekick/install.ps1 | iex
```

See [sidekick/INSTALL.md](sidekick/INSTALL.md) for prerequisites, configuration, and uninstall.

## Packages

| Package | Description |
|---------|-------------|
| [sidekick/](sidekick/) | MCP server — real-time meeting co-pilot for GitHub Copilot |
| [sidekick-notify/](sidekick-notify/) | VS Code extension — notification feed from Sidekick |

## License

Released under the [MIT License](LICENSE).

