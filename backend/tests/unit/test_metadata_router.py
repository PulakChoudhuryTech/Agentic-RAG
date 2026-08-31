from backend.app.rag.metadata_router import detect_metadata_filters


def test_detects_india_country_and_hr_category():
    filters = detect_metadata_filters("What is the parental leave policy in India?")
    assert filters == {"category": "hr", "country": "IN"}


def test_detects_it_category_from_vpn_keyword():
    filters = detect_metadata_filters("My VPN isn't working, help!")
    assert filters == {"category": "it"}


def test_detects_travel_category():
    filters = detect_metadata_filters("What is the hotel expense limit for travel?")
    assert filters == {"category": "travel"}


def test_no_match_returns_empty_dict():
    filters = detect_metadata_filters("Hello, how are you today?")
    assert filters == {}


def test_word_boundary_avoids_false_positive_substring_match():
    # "uk" should not match inside an unrelated word like "truck"
    filters = detect_metadata_filters("My truck broke down on the way to the airport for my flight")
    assert filters.get("country") is None


def test_united_states_phrase_maps_to_us():
    filters = detect_metadata_filters("What is the PTO policy in the United States?")
    assert filters["country"] == "US"
