from rag_common.secrets import REDACTED, looks_secret, redact_secrets


def test_redaction_catches_credentials_and_leaves_ordinary_text_alone():
    leaks = {
        "password hunter2": "hunter2",
        "pwd=abc123": "abc123",
        "my password is sunshine": "sunshine",
        "postgres://admin:S3cretPass@db.local:5432/x": "S3cretPass",
        "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n": "eyJhbGci",
        "key sk-AbCdEf1234567890xyzQ": "sk-AbCd",
        "pin: 4821": "4821",
        "here: q7Rk2MvX9pLw4NzT8bYc3HdJ6fGs1A": "q7Rk2MvX",
    }
    for text, secret in leaks.items():
        assert secret not in redact_secrets(text) and REDACTED in redact_secrets(text), text
    for text in ("Annual_Report_2024_Final_Version_Q3_Summary.pdf", "order ORD-2024-0000123456-ABCDEF-998877",
                 "the pin is bent and the token is rejected", "password policy requires 12 characters",
                 "doc 3f2a9c0b7e1d4a5f8c6b2e9d0a1f3c7b", "call getUserAccountBalanceForQuarter2024"):
        assert redact_secrets(text) == text, text
    assert looks_secret("my api key: abc") and not looks_secret("what does the report say about token usage?")
