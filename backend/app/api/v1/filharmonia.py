"""API endpoints for filharmonia.pl concert program scraping."""

import logging
from fastapi import APIRouter, HTTPException, Query

from app.services.filharmonia_scraper import (
    ConcertProgram,
    get_filharmonia_scraper,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/filharmonia", tags=["filharmonia"])


@router.get("/concerts", response_model=list[ConcertProgram])
def search_concerts(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    """Search for concerts on a given date."""
    scraper = get_filharmonia_scraper()
    try:
        concerts = scraper.search_concerts(date)
        return concerts
    except Exception as e:
        logger.error(f"Error searching concerts for {date}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search concerts: {str(e)}")


@router.get("/concert-details", response_model=ConcertProgram)
def get_concert_details(url: str = Query(..., description="Concert URL path, e.g. /repertuar/slug-123")):
    """Get detailed program for a specific concert."""
    scraper = get_filharmonia_scraper()
    try:
        details = scraper.get_concert_details(url)
        if not details:
            raise HTTPException(status_code=404, detail="Concert not found or could not be parsed")
        return details
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting concert details for {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get concert details: {str(e)}")
