"""HTTP round-trip tests for the dataset column-update contract (D9)."""

import uuid

import pytest

CSV = "age,gender,score\n25,Male,85.5\n30,Female,92.3\n22,Female,78.1\n"


@pytest.fixture(autouse=True)
def _tmp_storage(tmp_path, monkeypatch):
    from app.services.storage import storage_service

    monkeypatch.setattr(storage_service, "base_path", str(tmp_path))


async def _setup_dataset(client):
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"ds-{uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Dataset User",
            "expertise_level": "researcher",
        },
    )
    headers = {"Authorization": f"Bearer {res.json()['access_token']}"}
    proj = await client.post(
        "/api/v1/projects", json={"title": "Dataset Project"}, headers=headers
    )
    project_id = proj.json()["id"]

    upload = await client.post(
        f"/api/v1/projects/{project_id}/datasets",
        files={"file": ("data.csv", CSV, "text/csv")},
        headers=headers,
    )
    assert upload.status_code == 201, upload.text
    return headers, upload.json()["id"]


@pytest.mark.asyncio
async def test_columns_payload_changes_server_state(client):
    """The exact payload the frontend sends must actually persist role changes."""
    headers, dataset_id = await _setup_dataset(client)

    payload = {
        "columns": [
            {
                "name": "age",
                "dtype": "numeric",
                "role": "independent",
                "missing_count": 0,
                "missing_pct": 0.0,
                "unique_count": 3,
                "sample_values": ["25", "30", "22"],
            },
            {
                "name": "gender",
                "dtype": "categorical",
                "role": "control",
                "missing_count": 0,
                "missing_pct": 0.0,
                "unique_count": 2,
                "sample_values": ["Male", "Female"],
            },
            {
                "name": "score",
                "dtype": "numeric",
                "role": "dependent",
                "missing_count": 0,
                "missing_pct": 0.0,
                "unique_count": 3,
                "sample_values": ["85.5", "92.3", "78.1"],
            },
        ]
    }

    res = await client.put(
        f"/api/v1/datasets/{dataset_id}", json=payload, headers=headers
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["id"] == dataset_id
    assert body["status"] == "updated"
    assert body["applied_column_updates"] == 3

    detail = await client.get(f"/api/v1/datasets/{dataset_id}", headers=headers)
    assert detail.status_code == 200
    roles = {c["name"]: c.get("role") for c in detail.json()["columns"]}
    assert roles == {
        "age": "independent",
        "gender": "control",
        "score": "dependent",
    }


@pytest.mark.asyncio
async def test_no_op_columns_payload_reports_unchanged(client):
    headers, dataset_id = await _setup_dataset(client)

    res = await client.put(
        f"/api/v1/datasets/{dataset_id}",
        json={"columns": [{"name": "age", "dtype": "numeric"}]},
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json()["status"] == "unchanged"
    assert res.json()["applied_column_updates"] == 0


@pytest.mark.asyncio
async def test_legacy_column_updates_payload_still_works(client):
    headers, dataset_id = await _setup_dataset(client)

    res = await client.put(
        f"/api/v1/datasets/{dataset_id}",
        json={
            "column_updates": [
                {"original_name": "gender", "role": "participant_id"}
            ]
        },
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json()["applied_column_updates"] == 1

    detail = await client.get(f"/api/v1/datasets/{dataset_id}", headers=headers)
    roles = {c["name"]: c.get("role") for c in detail.json()["columns"]}
    assert roles["gender"] == "participant_id"


@pytest.mark.asyncio
async def test_update_unknown_dataset_returns_404(client):
    headers, _ = await _setup_dataset(client)
    res = await client.put(
        f"/api/v1/datasets/{uuid.uuid4()}",
        json={"columns": [{"name": "age", "role": "control"}]},
        headers=headers,
    )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_rename_collision_returns_400(client):
    headers, dataset_id = await _setup_dataset(client)
    res = await client.put(
        f"/api/v1/datasets/{dataset_id}",
        json={"column_updates": [{"original_name": "age", "name": "score"}]},
        headers=headers,
    )
    assert res.status_code == 400
    assert "score" in res.json()["detail"]


@pytest.mark.asyncio
async def test_cleaning_writes_cleaned_parquet_and_is_read_back(client):
    """Deletes, renames and cell edits must reach the dataframe the analyses read."""
    headers, dataset_id = await _setup_dataset(client)

    res = await client.put(
        f"/api/v1/datasets/{dataset_id}",
        json={
            "column_updates": [
                {"original_name": "gender", "deleted": True},
                {"original_name": "score", "name": "final_score"},
            ],
            "data_edits": [
                {"row_index": 0, "column_name": "age", "new_value": 26},
            ],
        },
        headers=headers,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["applied_column_updates"] == 2
    assert body["applied_data_edits"] == 1
    assert body["cleaned_parquet_path"].endswith("cleaned.parquet")
    assert [c["name"] for c in body["columns"]] == ["age", "final_score"]

    detail = (
        await client.get(f"/api/v1/datasets/{dataset_id}", headers=headers)
    ).json()
    assert [c["name"] for c in detail["columns"]] == ["age", "final_score"]
    first_row = detail["preview"][0]
    assert "gender" not in first_row
    assert first_row["age"] == 26
    assert first_row["final_score"] == 85.5
    assert detail["row_count"] == 3


@pytest.mark.asyncio
async def test_cleaning_is_cumulative_across_updates(client):
    headers, dataset_id = await _setup_dataset(client)

    await client.put(
        f"/api/v1/datasets/{dataset_id}",
        json={"data_edits": [{"row_index": 1, "column_name": "age", "new_value": 31}]},
        headers=headers,
    )
    await client.put(
        f"/api/v1/datasets/{dataset_id}",
        json={"column_updates": [{"original_name": "gender", "deleted": True}]},
        headers=headers,
    )

    detail = (
        await client.get(f"/api/v1/datasets/{dataset_id}", headers=headers)
    ).json()
    assert [c["name"] for c in detail["columns"]] == ["age", "score"]
    assert detail["preview"][1]["age"] == 31


@pytest.mark.asyncio
async def test_role_only_update_does_not_write_a_cleaned_parquet(client):
    headers, dataset_id = await _setup_dataset(client)

    res = await client.put(
        f"/api/v1/datasets/{dataset_id}",
        json={"columns": [{"name": "age", "role": "dependent"}]},
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json()["applied_column_updates"] == 1
    assert res.json()["cleaned_parquet_path"] is None


@pytest.mark.asyncio
async def test_invalid_data_edit_returns_400(client):
    headers, dataset_id = await _setup_dataset(client)

    bad_column = await client.put(
        f"/api/v1/datasets/{dataset_id}",
        json={
            "data_edits": [
                {"row_index": 0, "column_name": "nope", "new_value": 1}
            ]
        },
        headers=headers,
    )
    assert bad_column.status_code == 400
    assert "nope" in bad_column.json()["detail"]

    bad_row = await client.put(
        f"/api/v1/datasets/{dataset_id}",
        json={
            "data_edits": [
                {"row_index": 99, "column_name": "age", "new_value": 1}
            ]
        },
        headers=headers,
    )
    assert bad_row.status_code == 400
    assert "99" in bad_row.json()["detail"]


@pytest.mark.asyncio
async def test_dataset_detail_exposes_preview_rows_alias(client):
    headers, dataset_id = await _setup_dataset(client)

    detail = (
        await client.get(f"/api/v1/datasets/{dataset_id}", headers=headers)
    ).json()
    assert detail["preview_rows"] == detail["preview"]
    assert len(detail["preview_rows"]) == 3
    assert detail["column_count"] == 3
