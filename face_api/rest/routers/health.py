from __future__ import annotations

from fastapi import APIRouter, Request

from face_api.rest.responses import api_response
# 创建路由容器
router = APIRouter()

# 请求 GET http://localhost:8000/healthz时，返回健康状态，API进程是否还活着
@router.get("/healthz")
def healthz() -> dict:
    return api_response({"status": "ok"})

# 定义ready检查接口，检查程序是否能工作
@router.get("/readyz")
def readyz(request: Request) -> dict:
    # 获取存储服务
    store = request.app.state.store
    # 获取识别服务
    recognition = request.app.state.recognition_service
    try:
        # 获取运行环境信息，展示模型运行状态
        runtime = recognition.describe()
        # 获取人脸库数量；这是运营统计，不参与特征匹配兼容性判断。
        subjects = len(store.list_subjects())
        ready = True
    except Exception as exc:
        subjects = 0
        runtime = {"error": str(exc)}
        ready = False
    return api_response({"ready": ready, "subjects": subjects, "runtime": runtime})
