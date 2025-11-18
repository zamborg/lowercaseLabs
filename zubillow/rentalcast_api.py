"""
Thin wrapper around RentCast's *Rental Listings (Long Term)* endpoint.

Reference documentation:
    https://developers.rentcast.io/reference/rental-listings-long-term

This module intentionally mirrors the query parameters documented by RentCast so
that the resulting client can be used directly as a tool for LLM agents or in
small scripts. The API is very simple: instantiate :class:`RentCastClient`,
construct a :class:`ListingSearchParams`, and call :meth:`RentCastClient.search_listings`.

Example
-------
    from rentalcast_api import RentCastClient, ListingSearchParams

    client = RentCastClient()  # reads RENTCAST_API_KEY from the environment
    params = ListingSearchParams(city="Austin", state="TX", status="Active", limit=5)
    result = client.search_listings(params)
"""

from __future__ import annotations

import os
from typing import Any, Mapping
from urllib.parse import urlencode

import requests
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "RentCastClient",
    "RentCastError",
    "ListingSearchParams",
]

BASE_URL = "https://api.rentcast.io/v1/listings/rental/long-term"
DEFAULT_TIMEOUT = 15.0


class RentCastError(RuntimeError):
    """Raised when the RentCast API responds with an error."""


class ListingSearchParams(BaseModel):
    """
    Query parameters for the RentCast long-term rental listings endpoint.

    Field names are exposed in snake_case for Python ergonomics but map directly
    to the camelCase names that RentCast expects. Every parameter listed in the
    public documentation is included here, with type hints that roughly match
    the documented behaviour.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # Location filters
    address: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = Field(default=None, alias="postalCode")
    zip_code: str | None = Field(default=None, alias="zipCode")
    county: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    radius: float | None = None

    # Listing attributes
    status: str | None = None  # e.g. Active, Pending, Closed
    property_type: str | None = Field(default=None, alias="propertyType")
    listing_type: str | None = Field(default=None, alias="listingType")
    listing_source: str | None = Field(default=None, alias="listingSource")

    # Pricing filters
    min_price: int | None = Field(default=None, alias="minPrice")
    max_price: int | None = Field(default=None, alias="maxPrice")
    min_price_per_sqft: float | None = Field(default=None, alias="minPricePerSqft")
    max_price_per_sqft: float | None = Field(default=None, alias="maxPricePerSqft")
    min_previous_price: int | None = Field(default=None, alias="minPreviousPrice")
    max_previous_price: int | None = Field(default=None, alias="maxPreviousPrice")
    min_rent_price: int | None = Field(default=None, alias="minRentPrice")
    max_rent_price: int | None = Field(default=None, alias="maxRentPrice")

    # Bedroom / bathroom counts
    min_beds: float | None = Field(default=None, alias="minBeds")
    max_beds: float | None = Field(default=None, alias="maxBeds")
    min_baths: float | None = Field(default=None, alias="minBaths")
    max_baths: float | None = Field(default=None, alias="maxBaths")

    # Size filters
    min_sqft: int | None = Field(default=None, alias="minSqft")
    max_sqft: int | None = Field(default=None, alias="maxSqft")
    min_lot_size: int | None = Field(default=None, alias="minLotSize")
    max_lot_size: int | None = Field(default=None, alias="maxLotSize")
    min_garage_spaces: int | None = Field(default=None, alias="minGarageSpaces")
    max_garage_spaces: int | None = Field(default=None, alias="maxGarageSpaces")
    min_stories: float | None = Field(default=None, alias="minStories")
    max_stories: float | None = Field(default=None, alias="maxStories")
    min_year_built: int | None = Field(default=None, alias="minYearBuilt")
    max_year_built: int | None = Field(default=None, alias="maxYearBuilt")

    # Miscellaneous feature toggles
    has_pool: bool | None = Field(default=None, alias="hasPool")
    has_fireplace: bool | None = Field(default=None, alias="hasFireplace")
    has_basement: bool | None = Field(default=None, alias="hasBasement")
    has_garage: bool | None = Field(default=None, alias="hasGarage")
    has_air_conditioning: bool | None = Field(default=None, alias="hasAirConditioning")
    has_laundry: bool | None = Field(default=None, alias="hasLaundry")
    has_security_system: bool | None = Field(default=None, alias="hasSecuritySystem")
    has_solar: bool | None = Field(default=None, alias="hasSolar")
    has_waterfront: bool | None = Field(default=None, alias="hasWaterfront")
    has_view: bool | None = Field(default=None, alias="hasView")
    has_new_construction: bool | None = Field(default=None, alias="hasNewConstruction")
    include_sold: bool | None = Field(default=None, alias="includeSold")
    include_pending: bool | None = Field(default=None, alias="includePending")
    include_contingent: bool | None = Field(default=None, alias="includeContingent")

    # Temporal filters
    listed_after: str | None = Field(default=None, alias="listedAfter")  # ISO-8601 date string
    listed_before: str | None = Field(default=None, alias="listedBefore")
    pending_after: str | None = Field(default=None, alias="pendingAfter")
    pending_before: str | None = Field(default=None, alias="pendingBefore")
    closed_after: str | None = Field(default=None, alias="closedAfter")
    closed_before: str | None = Field(default=None, alias="closedBefore")
    withdrawn_after: str | None = Field(default=None, alias="withdrawnAfter")
    withdrawn_before: str | None = Field(default=None, alias="withdrawnBefore")
    expired_after: str | None = Field(default=None, alias="expiredAfter")
    expired_before: str | None = Field(default=None, alias="expiredBefore")
    updated_after: str | None = Field(default=None, alias="updatedAfter")
    updated_before: str | None = Field(default=None, alias="updatedBefore")
    created_after: str | None = Field(default=None, alias="createdAfter")
    created_before: str | None = Field(default=None, alias="createdBefore")
    days_on_market_min: int | None = Field(default=None, alias="minDaysOnMarket")
    days_on_market_max: int | None = Field(default=None, alias="maxDaysOnMarket")

    # Pagination & sorting
    cursor: str | None = None
    offset: int | None = None
    page: int | None = None
    limit: int | None = None
    sort_by: str | None = Field(default=None, alias="sortBy")
    sort_type: str | None = Field(default=None, alias="sortType")

    def to_query(self) -> dict[str, Any]:
        """
        Convert this parameter set into the dictionary expected by requests.
        """

        data = self.model_dump(by_alias=True, exclude_none=True)
        for key, value in list(data.items()):
            if isinstance(value, bool):
                data[key] = "true" if value else "false"
        return data


class RentCastClient:
    """
    Minimal client for the RentCast long-term rental listings API.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("RENTCAST_API_KEY")
        if not self.api_key:
            raise ValueError("RentCast API key not provided. Set RENTCAST_API_KEY or pass api_key explicitly.")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def search_listings(self, params: ListingSearchParams | Mapping[str, Any] | None = None) -> dict[str, Any]:
        """
        Call the `/listings/rental/long-term` endpoint.

        Parameters
        ----------
        params:
            Either a :class:`ListingSearchParams` instance or a mapping of query
            parameters. Only parameters with non-``None`` values are sent.
        """

        query: dict[str, Any] = {}
        if isinstance(params, ListingSearchParams):
            query = params.to_query()
        elif params:
            query = _compact_mapping(params)

        url = self.base_url
        if query:
            query_string = urlencode(query, doseq=True)
            url = f"{url}?{query_string}"


        response = self.session.get(
            url,
            headers={"X-Api-Key": self.api_key, "accept": "application/json"},
            timeout=self.timeout,
        )
        return response
        return self._parse_response(response)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _parse_response(self, response: requests.Response) -> dict[str, Any]:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - requires live API failure
            raise RentCastError(f"RentCast API error ({response.status_code}): {response.text}") from exc

        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - malformed JSON
            raise RentCastError("RentCast API returned an unreadable response.") from exc

        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"data": payload}
        return {"data": payload}


def _compact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Create a shallow copy of `mapping` without empty (None/\"\") values."""

    compact: dict[str, Any] = {}
    for key, value in mapping.items():
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            compact[key] = "true" if value else "false"
            continue
        compact[key] = value
    return compact
