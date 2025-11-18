"""
Manual chat runner that exposes RentCast helper tools to the OpenAI Responses API.

Usage
-----
    $ python manual_zillow.py

The CLI supports free-form chatting with the configured OpenAI model and also
lets the assistant call out to the RentCast API via JSON tool calls. For quick
manual inspection you can still issue commands yourself:

    /search city=Seattle state=WA min_beds=2
    /listing 123456789

Quit with `q` or `quit`.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from rentalcast_api import ListingSearchParams, RentCastClient, RentCastError

load_dotenv()

SYSTEM_PROMPT = """You are a meticulous rental search assistant helping a renter narrow down
options. Ask clarifying questions, reason about trade-offs, and keep track of useful
details that the renter mentions. When you reference RentCast data make sure you use
the latest information available."""

RENTCAST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rentcast_search_listings",
            "description": "Search RentCast long-term rental listings with rich filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name (e.g., Seattle)."},
                    "state": {"type": "string", "description": "Two-letter state code (e.g., WA)."},
                    "zip": {"type": "string", "description": "5-digit postal code."},
                    "latitude": {"type": "number", "description": "Latitude for point-radius search."},
                    "longitude": {"type": "number", "description": "Longitude for point-radius search."},
                    "radius": {"type": "number", "description": "Search radius in miles."},
                    "min_price": {"type": "integer", "description": "Minimum monthly rent (USD)."},
                    "max_price": {"type": "integer", "description": "Maximum monthly rent (USD)."},
                    "min_beds": {"type": "number", "description": "Minimum number of bedrooms."},
                    "max_beds": {"type": "number", "description": "Maximum number of bedrooms."},
                    "min_baths": {"type": "number", "description": "Minimum number of bathrooms."},
                    "max_baths": {"type": "number", "description": "Maximum number of bathrooms."},
                    "min_sqft": {"type": "integer", "description": "Minimum interior square footage."},
                    "max_sqft": {"type": "integer", "description": "Maximum interior square footage."},
                    "property_type": {
                        "type": "string",
                        "description": "Optional property type (e.g., single_family, condo, apartment).",
                    },
                    "limit": {"type": "integer", "description": "Number of results to return (max 100)."},
                    "include_sold": {
                        "type": "boolean",
                        "description": "Include inactive/sold listings (default false).",
                    },
                    "cursor": {"type": "string", "description": "Pagination cursor supplied by previous search."},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rentcast_get_listing",
            "description": "Fetch full details for a RentCast long-term rental listing by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_id": {
                        "type": "string",
                        "description": "Listing identifier provided by RentCast search results (listingId).",
                    }
                },
                "required": ["listing_id"],
                "additionalProperties": False,
            },
        },
    },
]


class Chat:
    """
    Minimal wrapper that keeps chat history, handles RentCast commands, executes
    tool calls, and forwards user queries to the OpenAI Responses API.
    """

    def __init__(
        self,
        *,
        llm_client: OpenAI | None = None,
        rentcast_client: RentCastClient | None = None,
        model: str = "gpt-5",
        reasoning: dict[str, str] | None = None,
    ) -> None:
        self.llm_client = llm_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rentcast_client = rentcast_client or _safe_create_rentcast_client()
        self.model = model
        self.reasoning = reasoning or {"effort": "medium", "summary": "brief"}
        self.previous_response_id: str | None = None
        self.messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print("Manual Zillow helper ready.")
        print("Commands: `/search key=value ...`, `/listing <listing_id>`, `q` to quit.")

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if not self.process_input(user_input):
                break

    def process_input(self, user_input: str) -> bool:
        """
        Handle a line of user input. Returns False when the chat session should end.
        """

        lowered = user_input.lower()
        if lowered in {"q", "quit"}:
            print("Goodbye!")
            return False

        if user_input.startswith("/search"):
            self._handle_search_command(user_input)
            return True

        if user_input.startswith("/listing"):
            self._handle_listing_command(user_input)
            return True

        self.chat_input(user_input)
        self.send(user_input)
        return True

    def chat_input(self, user_input: str) -> None:
        self.messages.append({"role": "user", "content": user_input})

    def send(self, user_input: str) -> None:
        kwargs = {
            "input": user_input,
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "reasoning": self.reasoning,
            "tools": RENTCAST_TOOLS,
        }
        if self.previous_response_id:
            kwargs["previous_response_id"] = self.previous_response_id

        response = self.llm_client.responses.create(**kwargs)  # pyright: ignore[reportPrivateImportUsage]
        self.previous_response_id = response.id
        self._handle_response(response)

    # ------------------------------------------------------------------ #
    # Response & tool handling
    # ------------------------------------------------------------------ #
    def _handle_response(self, response: Any) -> None:
        """
        Print assistant messages and fulfil tool calls until the response is complete.
        """

        # First, execute any required tool calls.
        while True:
            tool_calls = self._collect_tool_calls(response)
            if not tool_calls:
                break

            tool_outputs = []
            for call in tool_calls:
                output = self._execute_tool_call(call)
                tool_outputs.append({"tool_call_id": call["id"], "output": output})

            response = self._submit_tool_outputs(response.id, tool_outputs)
            if response is None:
                return

        # Print any assistant text emitted with the final response.
        content = _extract_text(response)
        if content:
            self.messages.append({"role": "assistant", "content": content})
            print(f"Agent: {content}")

    def _collect_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """
        Extract pending tool calls from the response object.
        """

        tool_calls: list[dict[str, Any]] = []
        required_action = getattr(response, "required_action", None)
        if required_action:
            submit = getattr(required_action, "submit_tool_outputs", None)
            if submit and getattr(submit, "tool_calls", None):
                for call in submit.tool_calls:
                    tool_calls.append(
                        {
                            "id": getattr(call, "id", None) or getattr(call, "tool_call_id", None),
                            "name": getattr(call, "function", {}).get("name")
                            if isinstance(getattr(call, "function", None), dict)
                            else getattr(getattr(call, "function", None), "name", None),
                            "arguments": getattr(call, "function", {}).get("arguments")
                            if isinstance(getattr(call, "function", None), dict)
                            else getattr(getattr(call, "function", None), "arguments", "{}"),
                        }
                    )

        # Fallback: inspect output blocks if required_action is not populated.
        output_blocks = getattr(response, "output", None)
        if output_blocks:
            for block in output_blocks:
                content = getattr(block, "content", None) or getattr(block, "tool_calls", None)
                if isinstance(content, list):
                    for item in content:
                        item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
                        if item_type != "tool_call":
                            continue
                        function = getattr(item, "function", None)
                        if isinstance(item, dict):
                            function = item.get("function", function)
                        arguments = "{}"
                        name = None
                        if isinstance(function, dict):
                            name = function.get("name")
                            arguments = function.get("arguments", "{}")
                        else:
                            name = getattr(function, "name", None)
                            arguments = getattr(function, "arguments", "{}")

                        tool_calls.append(
                            {
                                "id": getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None),
                                "name": name,
                                "arguments": arguments,
                            }
                        )
        return [call for call in tool_calls if call.get("name")]

    def _execute_tool_call(self, call: dict[str, Any]) -> str:
        """
        Run the local implementation for a tool call and return the JSON string output.
        """

        name = call.get("name")
        argument_payload = call.get("arguments") or "{}"
        try:
            arguments = json.loads(argument_payload)
        except json.JSONDecodeError:
            arguments = {}

        if name == "rentcast_search_listings":
            return self._tool_search_listings(arguments)
        if name == "rentcast_get_listing":
            return self._tool_get_listing(arguments)

        return json.dumps({"error": f"Unknown tool {name!r}"})

    def _tool_search_listings(self, arguments: dict[str, Any]) -> str:
        if self.rentcast_client is None:
            return json.dumps({"error": "RentCast API key missing."})

        normalised = _normalise_arguments(arguments)
        try:
            params = ListingSearchParams(**normalised)
        except TypeError as exc:
            return json.dumps({"error": f"Invalid search parameters: {exc}"})

        try:
            payload = self.rentcast_client.search_listings(params)
        except RentCastError as exc:
            return json.dumps({"error": f"RentCast search failed: {exc}"})

        data = payload.get("data", [])
        summaries = [_summarise_listing_dict(listing) for listing in data[:10]]
        result = {
            "total_results": payload.get("resultCount", len(data)),
            "returned": len(summaries),
            "cursor": payload.get("cursor"),
            "listings": summaries,
        }
        if len(data) > len(summaries):
            result["truncated"] = True
        return json.dumps(result, ensure_ascii=False)

    def _tool_get_listing(self, arguments: dict[str, Any]) -> str:
        if self.rentcast_client is None:
            return json.dumps({"error": "RentCast API key missing."})

        listing_id = arguments.get("listing_id") or arguments.get("id")
        if not listing_id:
            return json.dumps({"error": "Missing required argument 'listing_id'."})

        try:
            payload = self.rentcast_client.get_listing_by_id(str(listing_id))
        except RentCastError as exc:
            return json.dumps({"error": f"RentCast lookup failed: {exc}"})

        return json.dumps(payload, ensure_ascii=False)

    def _submit_tool_outputs(self, response_id: str, tool_outputs: list[dict[str, str]]) -> Any | None:
        """
        Submit tool outputs back to the Responses API, returning the updated response object.
        """

        try:
            return self.llm_client.responses.submit_tool_outputs(
                response_id=response_id,
                tool_outputs=tool_outputs,
            )
        except Exception as exc:  # pragma: no cover - network/API failure
            print(f"Failed to submit tool outputs: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # Manual RentCast commands
    # ------------------------------------------------------------------ #
    def _handle_search_command(self, command: str) -> None:
        if self.rentcast_client is None:
            print("RentCast API key missing. Set RENTCAST_API_KEY to enable searches.")
            return

        values = _parse_key_value_pairs(command[len("/search") :])
        try:
            params = ListingSearchParams(**values)
        except TypeError as exc:
            print(f"Invalid search parameters: {exc}")
            return

        try:
            payload = self.rentcast_client.search_listings(params)
        except RentCastError as exc:
            print(f"RentCast search failed: {exc}")
            return

        listings = payload.get("data", [])
        if not listings:
            print("No listings found for that query.")
            return

        count = payload.get("resultCount", len(listings))
        cursor = payload.get("cursor")
        print(f"Found {count} listing(s). Showing the first {min(len(listings), 5)}:")
        for listing in listings[:5]:
            print(f"- {_summarise_listing(listing)}")
        if cursor:
            print(f"More results available. Next cursor: {cursor}")

    def _handle_listing_command(self, command: str) -> None:
        if self.rentcast_client is None:
            print("RentCast API key missing. Set RENTCAST_API_KEY to enable listing lookups.")
            return

        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: /listing <listing_id>")
            return
        listing_id = parts[1].strip()
        if not listing_id:
            print("Usage: /listing <listing_id>")
            return

        try:
            payload = self.rentcast_client.get_listing_by_id(listing_id)
        except RentCastError as exc:
            print(f"RentCast lookup failed: {exc}")
            return

        print(f"Listing {listing_id}:")
        for key, value in payload.items():
            print(f"  {key}: {value}")


# --------------------------------------------------------------------------- #
# RentCast helper functions
# --------------------------------------------------------------------------- #

def _safe_create_rentcast_client() -> RentCastClient | None:
    try:
        return RentCastClient()
    except ValueError as exc:
        print(f"[RentCast disabled] {exc}")
        return None


def _parse_key_value_pairs(argument: str) -> dict[str, Any]:
    """
    Convert a space-separated list of key=value pairs into a dictionary.

    Keys accept snake_case (e.g. ``min_beds``) or the API's camelCase equivalents.
    Values are converted to integers/floats when they look numeric.
    """

    pairs = argument.strip().split()
    values: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            print(f"Ignoring fragment '{pair}' (expected key=value).")
            continue
        key, raw_value = pair.split("=", 1)
        normalised_key = _normalise_key(key)
        values[normalised_key] = _coerce_value(raw_value)
    return values


def _normalise_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Apply the same key normalisation used for manual CLI parameters.
    """

    return {_normalise_key(key): value for key, value in arguments.items()}


def _normalise_key(key: str) -> str:
    key = key.strip()
    if not key:
        return key

    canonical_map = {
        "minbeds": "min_beds",
        "maxbeds": "max_beds",
        "minbaths": "min_baths",
        "maxbaths": "max_baths",
        "minprice": "min_price",
        "maxprice": "max_price",
        "minsquarefeet": "min_sqft",
        "minsqft": "min_sqft",
        "maxsquarefeet": "max_sqft",
        "maxsqft": "max_sqft",
        "propertytype": "property_type",
        "includesold": "include_sold",
    }

    flattened = key.replace("_", "").replace("-", "").lower()
    return canonical_map.get(flattened, key.replace("-", "_"))


def _coerce_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _summarise_listing(listing: dict[str, Any]) -> str:
    address_parts = [
        listing.get("formattedAddress") or "Unknown address",
        listing.get("city"),
        listing.get("state"),
    ]
    address = ", ".join(part for part in address_parts if part)
    price = listing.get("listPrice") or listing.get("price") or "N/A"
    beds = listing.get("beds", "?")
    baths = listing.get("baths", "?")
    property_type = listing.get("propertyType", "property")
    return f"{address} | ${price}/mo | {beds} bd / {baths} ba | {property_type}"


def _summarise_listing_dict(listing: dict[str, Any]) -> dict[str, Any]:
    """
    Produce a compact dictionary that highlights the most relevant listing fields.
    """

    return {
        "listing_id": listing.get("listingId"),
        "formatted_address": listing.get("formattedAddress"),
        "city": listing.get("city"),
        "state": listing.get("state"),
        "zip": listing.get("zip"),
        "price": listing.get("listPrice") or listing.get("price"),
        "beds": listing.get("beds"),
        "baths": listing.get("baths"),
        "property_type": listing.get("propertyType"),
        "year_built": listing.get("yearBuilt"),
        "square_feet": listing.get("livingArea"),
        "lot_size": listing.get("lotSize"),
        "url": listing.get("url") or listing.get("listingUrl"),
    }


# --------------------------------------------------------------------------- #
# OpenAI response helpers
# --------------------------------------------------------------------------- #

def _extract_text(response: Any) -> str:
    """Normalise an OpenAI responses payload into plain text for display."""

    if getattr(response, "output_text", None):
        return str(response.output_text)
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        fragments = []
        for item in content:
            text_value = getattr(item, "text", None) or getattr(item, "content", None) or item
            fragments.append(str(text_value))
        return "\n".join(fragments)
    return ""


def main() -> None:
    Chat().run()


if __name__ == "__main__":
    main()
