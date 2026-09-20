from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db
from app.models.user import User
from app.schemas.dataset import DatasetUpdateRequest
from app.services import dataset as dataset_service
from app.services import project as project_service

router = APIRouter(tags=["datasets"])


@router.post("/projects/{project_id}/datasets", status_code=201)
async def upload_dataset(
    project_id: UUID,
    file: UploadFile = File(...),
    sheet_name: str | None = Query(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)
    content = await file.read()
    try:
        result = await dataset_service.upload_dataset(
            db, project_id, user.id, file.filename, content, sheet_name
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.get("/projects/{project_id}/datasets")
async def list_datasets(
    project_id: UUID,
    page: int = Query(1, ge=1),
    limit: int = Query(200, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)
    return await dataset_service.list_datasets(
        db, project_id, user.id, page=page, limit=limit
    )


@router.get("/datasets/{dataset_id}")
async def get_dataset_detail(
    dataset_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await dataset_service.get_dataset_preview(db, dataset_id, user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")


@router.put("/datasets/{dataset_id}")
async def update_dataset(
    dataset_id: UUID,
    req: DatasetUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await dataset_service.update_dataset(
            db,
            dataset_id,
            user.id,
            req.normalized_column_updates(),
            req.data_edits,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await dataset_service.delete_dataset(db, dataset_id, user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")
