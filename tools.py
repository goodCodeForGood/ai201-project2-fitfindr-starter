import os
import re
from dotenv import load_dotenv
from groq import Groq
from utils.data_loader import load_listings

load_dotenv()


def _get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to .env")
    return Groq(api_key=api_key)


def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    listings = load_listings()

    keywords = set(re.findall(r"\b\w+\b", description.lower()))
    matches = []

    for item in listings:
        if max_price is not None and float(item["price"]) > max_price:
            continue

        if size:
            item_size = str(item.get("size", "")).lower()
            if size.lower() not in item_size:
                continue

        searchable = " ".join([
            str(item.get("title", "")),
            str(item.get("description", "")),
            str(item.get("category", "")),
            " ".join(item.get("style_tags", [])),
            str(item.get("brand", "")),
            " ".join(item.get("colors", [])),
        ]).lower()

        score = sum(1 for keyword in keywords if keyword in searchable)

        if score > 0:
            matches.append((score, item))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in matches]


def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    client = _get_groq_client()

    items = wardrobe.get("items", [])

    if items:
        wardrobe_text = "\n".join(
            f"- {item}" for item in items
        )
        prompt = f"""
You are a helpful fashion stylist.

New thrifted item:
{new_item}

User wardrobe:
{wardrobe_text}

Suggest 1-2 complete outfits using the new item and pieces from the wardrobe.
Name specific wardrobe pieces when possible. Keep the response concise and practical.
"""
    else:
        prompt = f"""
You are a helpful fashion stylist.

The user has this new thrifted item:
{new_item}

Their wardrobe is currently empty.

Give 1-2 complete outfit ideas using common clothing pieces they could pair
with the item. Explain the vibe briefly. Do not claim they own anything.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def create_fit_card(outfit: str, new_item: dict) -> str:
    """Generate a short, shareable outfit caption."""

    if not outfit or not outfit.strip():
        return "Cannot create a fit card because the outfit suggestion is missing."

    try:
        client = _get_groq_client()

        prompt = f"""
Create a short, casual social-media fit card for this thrifted outfit.

Item:
- Name: {new_item.get("title", "Unknown item")}
- Price: ${new_item.get("price", 0)}
- Platform: {new_item.get("platform", "Unknown")}

Outfit suggestion:
{outfit}

Write 2-4 sentences.
Mention the item name, price, and platform naturally once.
Describe the outfit vibe.
Make it sound like a real person's OOTD caption, not an advertisement.
Do not use a heading.
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.9,
        )

        result = response.choices[0].message.content

        if not result or not result.strip():
            return "The fit card could not be generated."

        return result.strip()

    except Exception as exc:
        return f"Could not create fit card: {exc}"