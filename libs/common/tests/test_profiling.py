"""Detected column patterns and the warnings for columns whose stored type would mislead a query."""

import pytest

from rag_common.profiling import detect, identifier_name, leading_zero_codes


@pytest.mark.parametrize("values, sql_type, name, pattern, warned", [
    (["19800104", "20241231", "19990315"], "date", "Begin Date", "a DATE (written as YYYYMMDD numbers in the file)", False),
    (["19800104", "20241231", "19990315"], "number", "Event", "dates written as YYYYMMDD numbers", True),
    (["2025-01-01T04:39:18.208Z", "2025-03-17T21:34:00.040Z"], "date", "time", "a TIMESTAMP in UTC (ISO 8601 with a time zone", False),
    (["2025-01-01T04:39:18Z", "2025-03-17T21:34:00+02:00"], "text", "time", "ISO 8601 timestamps as text", True),
    (["1/5/2024", "12/31/2023", "7/4/2024 10:00"], "text", "When", "dates as text (M/D/YYYY)", True),
    (["2024-01", "2024-02", "2023-12"], "text", "Month", "dates as text (YYYY-MM)", True),
    (["45292", "45293"], "date", "Day", "a DATE (Excel day numbers in the file)", False),
    (["02134", "02139", "10001"], "text", "Postal", "codes with leading zeros", False),
    (["441", "4411,4412", "722"], "text", "NAICS Code", "codes or identifiers", False),
    (["5.2%", "3.1%", "(1.0%)"], "number", "Change", "percentages (stored without the % sign", False),
    (["5.2 %", "about 3", "1.1%"], "text", "Change", "percentages as text", True),
    (["$1,200", "$35.50", "$7"], "number", "Price", "currency amounts (stored without the currency sign)", False),
    (["12 km", "5.5 km", "100 km", "3 km"], "text", "Distance", "numbers with the unit 'km'", True),
    (["12", "15", "n/a2", "18", "21", "30", "7"], "text", "Score", "mixed: 86% numbers, 14% text", True),
    (["1999", "2005", "2017"], "number", "Year", "years", False),
])
def test_patterns(values, sql_type, name, pattern, warned):
    found, warning = detect(values, sql_type, name=name)
    assert found.startswith(pattern) and bool(warning) is warned


def test_years_need_a_year_like_name_and_slashed_dates_say_which_order_or_that_it_is_unknown():
    """Any 1800-2100 integers were called years (elevations, rents), and every slashed date
    was taken for M/D/YYYY."""
    assert detect(["1999", "2005", "2017"], "number", name="Elevation (m)") == ("", "")
    assert detect(["1999", "2005", "2017"], "number", name="Fiscal year")[0] == "years"
    assert detect(["13/04/2024", "05/06/2024", "21/12/2024"], "text", name="When")[0] == "dates as text (D/M/YYYY)"
    assert detect(["04/13/2024", "06/05/2024"], "text", name="When")[0] == "dates as text (M/D/YYYY)"
    found, warning = detect(["03/04/2024", "05/06/2024", "11/12/2024"], "text", name="When")
    assert "ambiguous" in found and "%d/%m/%Y" in warning and "%m/%d/%Y" in warning


def test_values_that_all_read_as_numbers_are_never_codes():
    minus = chr(0x2212)
    assert detect(["1.2e-05", "3.4E-08", "2e-3", "0.5"], "text", name="p")[0] != "codes or identifiers"
    assert detect([minus + "3.4", "4.5", minus + "10.2"], "text", name="Change")[0] != "codes or identifiers"
    assert detect(["+9.5%", "-0.4%", "+10.2%"], "number", name="Change")[0].startswith("percentages")
    assert detect(["A12", "B7", "C3", "D4"], "text", name="x")[0] == "codes or identifiers"


def test_a_unique_text_column_is_a_key_but_distinct_numbers_are_not():
    names = [f"State {i}" for i in range(57)]
    assert detect(names, "text", filled=57, distinct=57)[0] == "unique values (a key: one row each)"
    populations = [str(588753 + i * 1013) for i in range(57)]
    assert detect(populations, "number", filled=57, distinct=57) == ("", "")
    assert detect(names[:10], "text") == ("", "")  # without the column's counts no key is claimed


def test_warnings_say_how_to_query_the_column():
    assert "STRPTIME(CAST(col AS VARCHAR), '%Y%m%d')" in detect(["20240105", "20240317"], "number")[1]
    assert "TRY_CAST(regexp_extract(col" in detect(["12 km", "5 km", "7 km"], "text")[1]


def test_zero_padded_small_numbers_are_not_codes():
    assert not leading_zero_codes(["01", "02", "12"], "Month") and not leading_zero_codes(["00", "07", "23"], "Hour")
    assert not leading_zero_codes(["01", "05", "10"], "Bin")  # two digits at most: a zero-padded number
    assert not leading_zero_codes(["001", "015", "120"], "Day of year")  # a day count, by its name
    assert leading_zero_codes(["02134", "10001"], "Postal") and leading_zero_codes(["0012", "0340"], "Bin")
    assert detect(["01", "02", "12"], "number", name="Month") == ("", "")  # stored as numbers, no pattern


def test_dates_in_several_formats_and_with_day_and_comma():
    pattern, warning = detect(["2024-01-05", "01/06/2024", "Jan 7, 2024", "2024-01-08"], "text", name="When")
    assert pattern == "dates as text in 3 formats (YYYY-MM-DD, M/D/YYYY, Mon D, YYYY)"
    assert warning.endswith("TRY_STRPTIME(col, ['%Y-%m-%d', '%m/%d/%Y', '%b %d, %Y'])")
    assert detect(["Jan 7, 2024", "Feb 10, 2024"], "text")[0] == "dates as text (Mon D, YYYY)"


def test_identifier_names_but_not_counts_of_them():
    assert identifier_name("NAICS Code") and identifier_name("Product ID") and identifier_name("Zip code (5 digits)")
    assert not identifier_name("Codes issued") and not identifier_name("Number of IDs") and not identifier_name("Idaho")
