import io


def test_library_shows_metadata_and_supports_search_and_delete(library):
    client = library()
    docs = client.get("/api/documents").get_json()
    assert len(docs) == 8 and all(d["status"] == "processed" and d["owner"] == "Ana" for d in docs)
    contract = next(d for d in docs if d["doc_type"] == "contract" and d["meta"].get("policy_number") == "INS-001")
    assert contract["meta"]["end_date"] == "2027-01-04"

    assert [d["id"] for d in client.get("/api/documents?type=contract").get_json()] == [
        d["id"] for d in docs if d["doc_type"] == "contract"]
    by_text = client.get("/api/documents?q=cosmetic").get_json()  # text inside the document, not only its name
    assert by_text and all(d["doc_type"] == "contract" for d in by_text)

    assert client.get(f"/api/documents/{contract['id']}/file").mimetype == "application/pdf"
    assert client.delete(f"/api/documents/{contract['id']}").status_code == 200
    assert client.get(f"/api/documents/{contract['id']}").status_code == 404


def test_duplicate_upload_is_recognised_and_not_stored_twice(library, tmp_path):
    client = library(count=2)
    pdf = sorted((tmp_path / "docs" / "clean").glob("*.pdf"))[0]
    again = client.post("/api/documents", data={"files": (io.BytesIO(pdf.read_bytes()), "renamed.pdf")},
                        content_type="multipart/form-data").get_json()[0]
    assert again["is_new"] is False
    assert len(client.get("/api/documents").get_json()) == 2


def test_rejects_unsupported_files_and_unknown_resources(client):
    client.sign_up()
    res = client.post("/api/documents", data={"files": (io.BytesIO(b"not a pdf"), "x.pdf")},
                      content_type="multipart/form-data")
    assert "unsupported" in res.get_json()[0]["error"]
    assert client.get("/api/documents/12345/file").status_code == 404
    assert client.get("/api/conversations/999").status_code == 404
    assert client.post("/api/conversations/999/messages", json={"question": "hi"}).status_code == 404


def test_server_errors_do_not_leak_details(client, monkeypatch):
    client.sign_up()
    import src.chat as chat

    monkeypatch.setattr(chat, "list_conversations", lambda *a, **k: 1 / 0)
    client.raw.application.testing = False
    client.raw.application.config["PROPAGATE_EXCEPTIONS"] = False
    res = client.get("/api/conversations")
    assert res.status_code == 500 and "division" not in res.get_data(as_text=True)
