## 测试rest_api：

```bash
uv run python scripts/test_rest_api_interfaces.py # 测试rest_api

```

客户端控制台日志

![image-20260807163252043](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260807163252043.png)

服务端控制台日志

![image-20260807163314471](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260807163314471.png)

连通性测试

```
========== REQUEST ==========
GET /healthz HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

========== RESPONSE ==========
HTTP/1.1 200 OK
Content-Type: application/json

{
  "code": 0,
  "msg": "success",
  "data": {
    "status": "ok"
  }
}

========== REQUEST ==========
GET /readyz HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

========== RESPONSE ==========
HTTP/1.1 200 OK
Content-Type: application/json

{
  "code": 0,
  "msg": "success",
  "data": {
    "ready": true,
    "subjects": 1,
    "runtime": {
      "provider_mode": "auto",
      "detector": {
        "detector": "YOLO_ONNXRuntime",
        "providers": [
          "CUDAExecutionProvider",
          "CPUExecutionProvider"
        ]
      },
      "engine": {
        "feature_engine": "CVLface_ONNXRuntime",
        "recognition_model_path": "/app/models/onnx/cvlface_adaface_vit_base_kprpe_webface4m.onnx",
        "aligner_model_path": "/app/models/onnx/cvlface_dfa_mobilenet.onnx",
        "provider_mode": "auto",
        "providers": [
          "CUDAExecutionProvider",
          "CPUExecutionProvider"
        ],
        "face_margin": 0.55,
        "align_score_threshold": 0.0,
        "uses_keypoints": true
      }
    }
  }
}
```

注册人脸

```
========== REQUEST ==========
POST /api/recognition/subjects HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "subject_id": "rest_e2e_b1e201a9_001",
  "name": "rest-test-subject-1",
  "image_base64": "XXXXXX"
}

========== RESPONSE ==========
HTTP/1.1 201 Created
Content-Type: application/json

{
  "code": 0,
  "msg": "人脸录入成功",
  "data": {
    "subject_id": "rest_e2e_b1e201a9_001",
    "create_time": "2026-08-07 08:15:04"
  }
}

========== REQUEST ==========
POST /api/recognition/subjects HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "subject_id": "rest_e2e_b1e201a9_001",
  "name": "rest-test-subject-1",
  "image_base64": "XXXXXX"
}

========== RESPONSE ==========
HTTP/1.1 409 Conflict
Content-Type: application/json

{
  "code": 1005,
  "msg": "Subject already exists",
  "data": {}
}

========== REQUEST ==========
POST /api/recognition/subjects HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "subject_id": "rest_e2e_b1e201a9_002",
  "name": "rest-test-subject-2",
  "image_base64": "XXXXXX"
}

========== RESPONSE ==========
HTTP/1.1 201 Created
Content-Type: application/json

{
  "code": 0,
  "msg": "人脸录入成功",
  "data": {
    "subject_id": "rest_e2e_b1e201a9_002",
    "create_time": "2026-08-07 08:15:04"
  }
}
```

查询更改信息

```
========== REQUEST ==========
GET /api/recognition/subjects/rest_e2e_b1e201a9_001 HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

========== RESPONSE ==========
HTTP/1.1 200 OK
Content-Type: application/json

{
  "code": 0,
  "msg": "success",
  "data": {
    "subject_id": "rest_e2e_b1e201a9_001",
    "name": "rest-test-subject-1",
    "image_base64": "XXXXXX",
    "create_time": "2026-08-07 08:15:04",
    "update_time": "2026-08-07 08:15:04"
  }
}

========== REQUEST ==========
PUT /api/recognition/subjects/rest_e2e_b1e201a9_001 HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "name": "rest-test-subject-1-updated"
}

========== RESPONSE ==========
HTTP/1.1 200 OK
Content-Type: application/json

{
  "code": 0,
  "msg": "主体信息更新成功",
  "data": {
    "subject_id": "rest_e2e_b1e201a9_001",
    "update_time": "2026-08-07 08:15:04"
  }
}

========== REQUEST ==========
GET /api/recognition/subjects/rest_e2e_b1e201a9_001 HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

========== RESPONSE ==========
HTTP/1.1 200 OK
Content-Type: application/json

{
  "code": 0,
  "msg": "success",
  "data": {
    "subject_id": "rest_e2e_b1e201a9_001",
    "name": "rest-test-subject-1-updated",
    "image_base64": "XXXXXX",
    "create_time": "2026-08-07 08:15:04",
    "update_time": "2026-08-07 08:15:04"
  }
}
```

人脸匹配识别

```
========== REQUEST ==========
POST /api/recognition/match HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "image_base64": "XXXXXX",
  "top_k": 2,
  "threshold": 0.0
}

========== RESPONSE ==========
HTTP/1.1 200 OK
Content-Type: application/json

{
  "code": 0,
  "msg": "比对完成",
  "data": {
    "result": [
      {
        "subject_id": "rest_e2e_b1e201a9_001",
        "name": "rest-test-subject-1-updated",
        "similarity": 0.4450995624065399
      },
      {
        "subject_id": "person_001",
        "name": "张三三",
        "similarity": 0.4449325203895569
      }
    ]
  }
}
```

图片格式错误+查询未注册人脸

```
========== REQUEST ==========
POST /api/recognition/match HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "image_base64": "not-a-valid-base64"
}

========== RESPONSE ==========
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "code": 1002,
  "msg": "图片 Base64 格式错误",
  "data": {}
}

========== REQUEST ==========
GET /api/recognition/subjects/not_existing_rest_e2e_subject HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

========== RESPONSE ==========
HTTP/1.1 404 Not Found
Content-Type: application/json

{
  "code": 1006,
  "msg": "Subject not found",
  "data": {}
}
```

删除注册人脸

```
========== REQUEST ==========
DELETE /api/recognition/subjects/rest_e2e_b1e201a9_002 HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

========== RESPONSE ==========
HTTP/1.1 200 OK
Content-Type: application/json

{
  "code": 0,
  "msg": "人脸数据删除成功",
  "data": {
    "subject_id": "rest_e2e_b1e201a9_002"
  }
}
```