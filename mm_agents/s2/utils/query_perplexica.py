import requests
import os

_cached_provider_ids = {}


def _get_provider_ids(perplexica_base: str) -> dict:
    """Fetch provider IDs from the Perplexica config endpoint and cache them."""
    if perplexica_base in _cached_provider_ids:
        return _cached_provider_ids[perplexica_base]

    config_url = perplexica_base.rstrip("/") + "/api/config"
    try:
        resp = requests.get(config_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        providers = data.get("values", {}).get("modelProviders", [])

        _PREFERRED_CHAT = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"]

        chat_provider_id = None
        embedding_provider_id = None
        chat_model_key = None
        embedding_model_key = None

        for p in providers:
            has_chat = bool(p.get("chatModels"))
            has_embed = bool(p.get("embeddingModels"))
            if has_chat and chat_provider_id is None:
                chat_provider_id = p["id"]
                model_keys = {m["key"] for m in p["chatModels"]}
                chat_model_key = next(
                    (k for k in _PREFERRED_CHAT if k in model_keys),
                    p["chatModels"][0]["key"],
                )
            if not has_chat and has_embed and embedding_provider_id is None:
                embedding_provider_id = p["id"]
                embedding_model_key = p["embeddingModels"][0]["key"]

        if embedding_provider_id is None:
            for p in providers:
                if p.get("embeddingModels"):
                    embedding_provider_id = p["id"]
                    embedding_model_key = p["embeddingModels"][0]["key"]
                    break

        result = {
            "chatProviderId": chat_provider_id,
            "chatModelKey": chat_model_key or "gpt-4o",
            "embeddingProviderId": embedding_provider_id,
            "embeddingModelKey": embedding_model_key or "text-embedding-3-small",
        }
        _cached_provider_ids[perplexica_base] = result
        return result
    except Exception as e:
        print(f"Failed to fetch Perplexica config: {e}")
        return {
            "chatProviderId": None,
            "chatModelKey": "gpt-4o",
            "embeddingProviderId": None,
            "embeddingModelKey": "text-embedding-3-small",
        }


def query_to_perplexica(query):
    base_url = os.getenv("PERPLEXICA_URL", "http://localhost:3000")
    perplexica_base = base_url.rstrip("/").replace("/api/search", "")
    search_url = perplexica_base + "/api/search"

    ids = _get_provider_ids(perplexica_base)

    message = {
        "query": query,
        "sources": ["web"],
        "optimizationMode": "balanced",
        "chatModel": {
            "providerId": ids["chatProviderId"],
            "key": ids["chatModelKey"],
        },
        "embeddingModel": {
            "providerId": ids["embeddingProviderId"],
            "key": ids["embeddingModelKey"],
        },
        "history": [],
        "stream": False,
    }

    try:
        response = requests.post(search_url, json=message, timeout=60)

        if response.status_code == 200:
            result = response.json()
            answer = result.get("message", "")
            sources = result.get("sources", [])

            formatted_response = answer
            if sources:
                formatted_response += "\n\nSources:\n"
                for i, source in enumerate(sources[:5], 1):
                    if isinstance(source, dict):
                        title = source.get("title", "")
                        url = source.get("url", "")
                        if title and url:
                            formatted_response += f"{i}. {title}: {url}\n"

            return formatted_response
        elif response.status_code == 400:
            print(f"Perplexica API error 400: {response.text}")
            return "Search request failed. Using model's internal knowledge."
        else:
            print(f"Perplexica API error {response.status_code}: {response.text}")
            return "Search service unavailable. Using model's internal knowledge."
    except requests.exceptions.Timeout:
        print("Perplexica request timeout")
        return "Search request timeout. Using model's internal knowledge."
    except requests.exceptions.ConnectionError:
        print("Cannot connect to Perplexica. Make sure it's running at:", search_url)
        return "Cannot connect to search service. Using model's internal knowledge."
    except Exception as e:
        print(f"Perplexica error: {e}")
        return "Search error. Using model's internal knowledge."


# Test Code
if __name__ == "__main__":
    query = "What is Agent S?"
    response = query_to_perplexica(query)
    print(response)
