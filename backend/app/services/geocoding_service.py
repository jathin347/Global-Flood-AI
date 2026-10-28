import httpx
from app.core.config import settings
from app.core.logging import logger
from app.schemas.location import LocationSearchResult, LocationSearchResponse

class GeocodingService:
    def __init__(self):
        self.timeout = settings.HTTP_TIMEOUT_SECONDS

    async def search(self, query: str, limit: int = 10) -> LocationSearchResponse:
        cleaned_query = query.strip()
        if not cleaned_query:
            return LocationSearchResponse(query=query, count=0, results=[])

        # Primary: Open-Meteo Geocoding
        try:
            params = {
                "name": cleaned_query,
                "count": min(limit, 20),
                "language": "en",
                "format": "json"
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(settings.OPEN_METEO_GEOCODING_URL, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_results = data.get("results", []) or []
                    results = []
                    for item in raw_results:
                        results.append(
                            LocationSearchResult(
                                id=item.get("id"),
                                name=item.get("name"),
                                latitude=item.get("latitude"),
                                longitude=item.get("longitude"),
                                country=item.get("country"),
                                country_code=item.get("country_code"),
                                admin1=item.get("admin1"),
                                elevation=item.get("elevation"),
                                timezone=item.get("timezone")
                            )
                        )
                    if results:
                        return LocationSearchResponse(query=query, count=len(results), results=results)
        except Exception as e:
            logger.warning(f"Open-Meteo geocoding error for '{query}': {e}. Trying fallback.")

        # Fallback: OpenStreetMap Nominatim
        try:
            headers = {"User-Agent": "GlobalFloodAI/1.0 (academic-research; contact@globalflood.ai)"}
            params = {
                "q": cleaned_query,
                "format": "json",
                "limit": limit,
                "addressdetails": 1
            }
            async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
                resp = await client.get("https://nominatim.openstreetmap.org/search", params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    results = []
                    for item in data:
                        addr = item.get("address", {})
                        results.append(
                            LocationSearchResult(
                                id=item.get("osm_id"),
                                name=item.get("display_name", "").split(",")[0],
                                latitude=float(item.get("lat")),
                                longitude=float(item.get("lon")),
                                country=addr.get("country"),
                                country_code=addr.get("country_code", "").upper() if addr.get("country_code") else None,
                                admin1=addr.get("state") or addr.get("county"),
                                elevation=None,
                                timezone=None
                            )
                        )
                    return LocationSearchResponse(query=query, count=len(results), results=results)
        except Exception as fallback_err:
            logger.error(f"Geocoding fallback failed for '{query}': {fallback_err}")

        return LocationSearchResponse(query=query, count=0, results=[])

geocoding_service = GeocodingService()
