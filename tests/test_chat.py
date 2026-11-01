def _ask(client, conv_id, question):
    res = client.post(f"/api/conversations/{conv_id}/messages", json={"question": question})
    assert res.status_code == 200, res.get_json()
    return res.get_json()["message"]


def test_conversation_answers_with_sources_and_resolves_follow_ups(library):
    client = library()
    conv = client.post("/api/conversations", json={}).get_json()["id"]

    first = _ask(client, conv, "What is the expiry date of contract INS-001?")
    assert "2027-01-04" in first["content"] and not first["refused"]
    assert first["sources"][0]["file_name"].startswith("contract") and first["sources"][0]["page"] == 1

    follow_up = _ask(client, conv, "And its annual ceiling?")  # no contract named: INS-001 is remembered
    assert "50 000.00 MAD" in follow_up["content"]
    assert {s["contract"] for s in follow_up["sources"]} == {"INS-001"}

    assert not _ask(client, conv, "Which contracts cover optical care?")["refused"]  # several contracts: no carry-over
    refused = _ask(client, conv, "Does any contract cover funeral expenses?")
    assert refused["refused"] and refused["sources"] == []


def test_conversations_persist_and_can_be_resumed_listed_and_deleted(library):
    client = library()
    conv = client.post("/api/conversations", json={}).get_json()["id"]
    assert client.get("/api/conversations").get_json() == []  # empty drafts do not clutter the history
    _ask(client, conv, "What is the expiry date of contract INS-001?")

    history = client.get("/api/conversations").get_json()
    assert [h["title"] for h in history] == ["What is the expiry date of contract INS-001?"]
    assert client.get("/api/conversations?q=expiry").get_json() and not client.get("/api/conversations?q=zzz").get_json()

    reopened = client.get(f"/api/conversations/{conv}").get_json()
    assert [m["role"] for m in reopened["messages"]] == ["user", "assistant"] and reopened["entities"][0] == "INS-001"
    assert "50 000.00 MAD" in _ask(client, conv, "What about its annual ceiling?")["content"]  # resumed later

    assert client.delete(f"/api/conversations/{conv}").status_code == 200
    assert client.get(f"/api/conversations/{conv}").status_code == 404


def test_conversation_scoped_to_one_document(library):
    client = library()
    docs = client.get("/api/documents").get_json()
    claim = next(d for d in docs if d["doc_type"] == "claim_form")
    conv = client.post("/api/conversations", json={"document_id": claim["id"]}).get_json()["id"]
    answer = _ask(client, conv, "What is the claimed amount?")
    assert answer["sources"] and {s["document_id"] for s in answer["sources"]} == {claim["id"]}
    assert client.get(f"/api/conversations/{conv}").get_json()["document"]["file_name"] == claim["file_name"]


def test_questions_about_the_selected_document_need_no_identifier(library):
    client = library()
    contract = next(d for d in client.get("/api/documents").get_json() if d["doc_type"] == "contract")
    conv = client.post("/api/conversations", json={"document_id": contract["id"]}).get_json()["id"]
    assert "20" in _ask(client, conv, "When does it expire?")["content"]  # a date, found with no contract number
    assert _ask(client, conv, "Which treatments are excluded?")["sources"]
    assert _ask(client, conv, "Is flight delay compensation covered?")["refused"]


def test_chats_can_be_renamed_and_pinned(library):
    client = library(count=4)
    first = client.post("/api/conversations", json={}).get_json()["id"]
    _ask(client, first, "What is the expiry date of contract INS-001?")
    second = client.post("/api/conversations", json={}).get_json()["id"]
    _ask(client, second, "What is the annual ceiling of contract INS-001?")
    assert [c["id"] for c in client.get("/api/conversations").get_json()] == [second, first]  # newest first

    assert client.patch(f"/api/conversations/{first}", json={"pinned": True}).get_json()["pinned"] is True
    listing = client.get("/api/conversations").get_json()
    assert [c["id"] for c in listing] == [first, second] and listing[0]["pinned"] and not listing[1]["pinned"]

    renamed = client.patch(f"/api/conversations/{second}", json={"title": "  Ceiling   check  "}).get_json()
    assert renamed["title"] == "Ceiling check" and renamed["pinned"] is False  # other fields untouched
    assert client.patch(f"/api/conversations/{second}", json={"title": "   "}).status_code == 400
    assert client.get(f"/api/conversations/{second}").get_json()["document"] is None  # scope not reset by the rename

    client.patch(f"/api/conversations/{first}", json={"pinned": False})
    assert [c["id"] for c in client.get("/api/conversations").get_json()] == [second, first]


def test_follow_up_detection():
    from src import chat

    assert chat.is_follow_up("And its annual ceiling?") and chat.is_follow_up("Which treatments are excluded?")
    assert not chat.is_follow_up("Which contracts cover optical care?")
    assert not chat.is_follow_up("What is the expiry date of contract INS-003?")


def test_model_citation_styles_are_normalised():
    from src.chat import normalise_citations

    assert normalise_citations("Ends 2027-01-04【1】 and 【2†contract_00.pdf】.") == "Ends 2027-01-04[1] and [2]."
    assert normalise_citations("Already [3] fine") == "Already [3] fine"
