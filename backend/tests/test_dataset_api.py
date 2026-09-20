def test_upload_endpoint_exists():
    """Verify the upload endpoint is registered."""
    from app.api.datasets import router
    routes = [r.path for r in router.routes]
    assert "/projects/{project_id}/datasets" in routes


def test_list_endpoint_exists():
    from app.api.datasets import router
    routes = [r.path for r in router.routes]
    assert "/projects/{project_id}/datasets" in routes


def test_detail_endpoint_exists():
    from app.api.datasets import router
    routes = [r.path for r in router.routes]
    assert "/datasets/{dataset_id}" in routes


def test_delete_endpoint_exists():
    from app.api.datasets import router
    routes = [r.path for r in router.routes]
    assert "/datasets/{dataset_id}" in routes


def test_update_endpoint_exists():
    from app.api.datasets import router
    routes = [r.path for r in router.routes]
    assert "/datasets/{dataset_id}" in routes
