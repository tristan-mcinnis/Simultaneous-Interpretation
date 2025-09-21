from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from openai import OpenAI


@dataclass(slots=True)
class OpenAITranslator:
    client: OpenAI
    model: str
    temperature: float = 0.0

    def translate(self, sentence: str, target_language: str, previous_chunks: Sequence[str], topic: str) -> str:
        previous_context = "\n".join(chunk for chunk in previous_chunks if chunk).strip()
        if not previous_context:
            previous_context = "None"
        topic_line = topic or "General conversation"
        user_prompt = (
            "Translate the following sentence into {language}. Return only the translation, no commentary.\n\n"
            "Sentence: {sentence}\n"
            "Previous Chunks: {previous}\n"
            "Topic: {topic}"
        ).format(language=target_language, sentence=sentence, previous=previous_context, topic=topic_line)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional simultaneous interpreter. "
                        "Focus on faithful, natural-sounding translations and maintain tone."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
