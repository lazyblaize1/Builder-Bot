# Builder-Bot

An AI-assisted Discord server management bot.

## Configuration

Copy `.env.example` to `.env` and set your Discord bot token and Groq API key
values. Never commit `.env` or share its contents. `GROQ_API_KEYS` accepts a
comma-separated list of keys; `GROQ_API_KEY` is available as a legacy
single-key fallback.

Keep your Discord bot token and Groq API keys private. If a key has been
exposed, revoke or rotate it with its provider before using it again.

## Running

Install the project dependencies from `requirements.txt`, then start the bot
with `python main.py`. The bot needs the appropriate Discord application
configuration and server permissions to function.

Independent category/channel creation and permission updates are run
concurrently with a bounded request pool. Discord 429 responses use the
library's rate-limit handling and an application-level retry/backoff for
rate-limit exceptions, while bulk actions continue and report individual
failures rather than aborting the batch.
