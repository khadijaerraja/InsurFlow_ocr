"""LLM clients for the answer-generation step.

Two implementations share one informal interface (`generate(prompt) -> str`):
`ScriptedLLM`, a test double that returns pre-queued responses so the
citation-bearing answer step can be tested with no network and no API key,
and `OpenAILLM`, an optional live client used only when OPENAI_API_KEY is set
and network access is available. Retrieval and the hallucination guard do not
depend on either -- both work from embedding similarity alone.
"""

from __future__ import annotations

import os
import time


class ScriptedLLM:
    """Test double returning queued canned responses instead of calling a real model."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("ScriptedLLM called with no queued responses left")
        return self._responses.pop(0)


class ExtractiveLLM:
    """No-model answerer: picks the line, across all retrieved excerpts, that best matches the question.

    Free, offline and honest: it can only repeat text that exists in the documents. It answers
    "expiry date of contract X" with the "Expiry date: ..." line and "exclusions"/"coverage"
    questions with that whole section, and cites the excerpt it took the text from.
    Works on the numbered prompt built by src/doc_qa.py (only the part after "Excerpts:").
    """

    _SECTIONS = {"exclu": ("EXCLUSIONS",), "guara": ("COVERAGE", "GARANTIES"), "cover": ("COVERAGE", "GARANTIES"),
                 "garan": ("GARANTIES", "COVERAGE"), "couve": ("GARANTIES", "COVERAGE")}
    _STOP = frozenset("what which who how when where does the and for are any under this that with from"
                      " tell about please its quel quelle quels quelles est les des une que qui quand combien comment"
                      " pour dans".split())
    _FOOTER = ("authorised", "date and", "synthetic", "page ")

    @staticmethod
    def _stem(word: str) -> str:
        return word[:5]

    @staticmethod
    def _pretty(line: str) -> str:
        """'Expiry.date: 2027-01-04' -> '**Expiry date:** 2027-01-04' (OCR artefacts tidied)."""
        import re

        label, sep, value = line.partition(":")
        if sep and value.strip():
            label = re.sub(r"[._]+", " ", label).strip()
            value = re.sub(r"(\d)\s*MAD", r"\1 MAD", value.strip())
            return f"**{label}:** {value}"
        return line

    @staticmethod
    def _known_term(item: str) -> tuple[str, str] | None:
        """(canonical name, remaining text) when the item starts with a known guarantee/exclusion.

        Scans often lose the spaces ("Cosmeticsurgery"), so the known name is shown instead."""
        from src.processing.extraction import EXCLUSION_TERMS, GUARANTEE_TERMS

        from src.retrieval.text import fold

        vocabulary = {**GUARANTEE_TERMS, **EXCLUSION_TERMS}
        squashed = ""
        for i, ch in enumerate(item):
            if ch.isalnum():
                squashed += fold(ch)
            for name, wordings in vocabulary.items():
                if squashed in wordings:
                    return name, item[i + 1:].strip()
        return None

    @classmethod
    def _section_items(cls, raw: list[str]) -> list[str]:
        """Clean the lines of a COVERAGE / EXCLUSIONS section into readable bullets."""
        out, i = [], 0
        while i < len(raw):
            known = cls._known_term(raw[i])
            if known is None:
                out.append(raw[i])
            else:
                name, rest = known
                if not rest and i + 1 < len(raw) and cls._known_term(raw[i + 1]) is None:
                    rest, i = raw[i + 1], i + 1  # the rate/limit was printed on the next line
                # keep the detail only when readable (a scan often glues the words together)
                readable = rest and not rest.startswith("(") and (rest.count(" ") >= 2 or rest.isalpha())
                out.append(f"{name} - {rest}" if readable else name)
            i += 1
        return out

    def generate(self, prompt: str) -> str:
        import re

        from src.retrieval.text import fold

        question = re.search(r"Question:\s*(.+?)\nAnswer:", prompt, re.S).group(1).strip()
        evidence = prompt.split("Excerpts:\n", 1)[-1]
        excerpts = re.findall(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\n\[\d+\]|\n\nQuestion:)", evidence, re.S)
        asked = {f"{p.upper()}-{int(n):03d}" for p, n in re.findall(r"\b([A-Za-z]{3})[-\s]?(\d{3,6})\b", question)}
        if asked:  # a question about INS-004 must be answered from an excerpt that mentions INS-004
            matching = [(n, b) for n, b in excerpts if any(i in b.upper() for i in asked)]
            excerpts = matching or excerpts
        # Identifiers (INS-001...) are in many lines and say nothing about which line answers the question.
        question_words = re.sub(r"\b[A-Za-z]{3}[-\s]?\d{3,6}\b", " ", question)
        words = [w for w in re.findall(r"[a-z]+", fold(question_words)) if w not in self._STOP and len(w) > 2]
        stems = {self._stem(w) for w in words}

        def score(line: str) -> int:
            """Shared words in the line's label, minus words the label has that the question does not
            ('New annual ceiling' loses to 'Annual ceiling' for 'what is the annual ceiling')."""
            label = {self._stem(w) for w in re.findall(r"[a-z]+", fold(line.partition(":")[0]))}
            shared = len(label & stems)
            return shared * 10 - len(label - stems) if shared else 0

        best = None  # (score, -excerpt_number, line_index, lines, number)
        for number, body in excerpts:
            lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
            for i, ln in enumerate(lines):
                if score(ln) > 0:
                    candidate = (score(ln), -int(number), i, lines, number)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
        if best is None:
            return "The retrieved passages do not contain a matching line. See the sources below."
        _, _, idx, lines, number = best

        upper = [ln.upper() for ln in lines]
        heading = next((h for s, hs in self._SECTIONS.items() if s in stems for h in hs if h in upper), None)
        if heading:
            raw = []
            for ln in lines[upper.index(heading) + 1:]:
                if (ln.isupper() and len(ln) > 3) or ln.lower().startswith(self._FOOTER):
                    break  # next section or the page footer
                raw.append(ln.lstrip("- ").strip())
            items = self._section_items(raw)
            if items:
                return f"**{heading.title()}** [{number}]\n" + "\n".join(f"- {item}" for item in items)
        answer = lines[idx]
        if answer.endswith(":") and idx + 1 < len(lines):
            answer = f"{answer} {lines[idx + 1]}"
        return f"{self._pretty(answer)} [{number}]"


class OpenAILLM:
    """Live OpenAI-backed client. Requires OPENAI_API_KEY and network access."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; live generation is unavailable")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""


class GroqLLM:
    """Groq's OpenAI-compatible chat API (free tier, no local model to download).
    Requires GROQ_API_KEY; model can be overridden with GROQ_MODEL."""

    def __init__(self, model: str | None = None) -> None:
        self._api_key = os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY is not set; live generation is unavailable")
        self._model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

    def generate(self, prompt: str) -> str:
        import json
        import urllib.request

        body = json.dumps(
            {"model": self._model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
        ).encode()
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            body,
            # Groq sits behind a firewall that rejects urllib's default User-Agent with a 403.
            {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}",
             "User-Agent": "InsurFlow/1.0 (+local prototype)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            text = json.loads(response.read())["choices"][0]["message"]["content"] or ""
        return text.replace("‑", "-").replace("‐", "-")  # typographic hyphens break copy/paste of dates


class OllamaLLM:
    """Local model served by Ollama (free, offline). Needs `ollama pull <model>` first."""

    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434") -> None:
        self._model = model
        self._url = f"{host}/api/generate"

    def generate(self, prompt: str) -> str:
        import json
        import urllib.request

        body = json.dumps(
            {"model": self._model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
        ).encode()
        request = urllib.request.Request(self._url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())["response"]


class GeminiLLM:
    """Live Gemini-backed client, alternative to OpenAILLM. Requires GEMINI_API_KEY."""

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        api_key = os.environ.get("")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; live generation is unavailable")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = _generate_with_retry(self._client, self._model, prompt)
        return response.text or ""


def _generate_with_retry(client, model: str, contents: str, attempts: int = 6):
    """Gemini free-tier keys are capped at ~5 requests/minute per model and
    occasionally return transient 503s under load; the SDK's own retry budget
    is too short to ride either out, so retry here with real backoff -- long
    enough to clear a per-minute quota window -- before giving up.
    """
    from google.genai.errors import ClientError, ServerError

    delay = 15.0
    for attempt in range(1, attempts + 1):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except ServerError:
            if attempt == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 60.0)
        except ClientError as exc:
            if getattr(exc, "code", None) != 429 or attempt == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 60.0)


def make_llm(name: str = "auto", model: str | None = None):
    """Returns (client, name). 'auto' uses Groq or Gemini when a key is configured, else the offline extractive answerer."""
    if name == "auto":
        name = "groq" if os.environ.get("GROQ_API_KEY") else "gemini" if os.environ.get("GEMINI_API_KEY") else "extractive"
    return {
        "extractive": ExtractiveLLM,
        "groq": lambda: GroqLLM(model),
        "gemini": lambda: GeminiLLM(model) if model else GeminiLLM(),
        "ollama": lambda: OllamaLLM(model or "llama3.2:3b"),
    }[name](), name
