

## 测试分析长视频

```bash
# 测试长视频分析
uv run python scripts/test_video_analysis_interfaces.py \
--video-source-url \
http://mirror.aarnet.edu.au/pub/TED-talks/911Mothers_2010W-480p.mp4

```

客户端日志

![image-20260807170731281](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260807170731281.png)

服务端日志

![image-20260807171612944](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260807171612944.png)

错误的url

```
========== REQUEST ==========
POST /api/recognition/video/analyze HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "task_name": "video_e2e_9c7e55cef480_invalid",
  "source_url": "not-a-valid-url"
}

========== RESPONSE ==========
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/json

{
  "code": 1001,
  "msg": "Invalid request argument",
  "data": {}
}
```

传入正确的url，创建分析任务，开始分析

```
========== REQUEST ==========
POST /api/recognition/video/analyze HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "task_name": "video_e2e_9c7e55cef480",
  "source_url": "http://mirror.aarnet.edu.au/pub/TED-talks/911Mothers_2010W-480p.mp4"
}

========== RESPONSE ==========
HTTP/1.1 201 Created
Content-Type: application/json

{
  "code": 0,
  "msg": "长视频分析任务创建成功",
  "data": {
    "task_id": "video_task_20260807_003",
    "task_name": "video_e2e_9c7e55cef480",
    "create_time": "2026-08-07 08:16:07",
    "source_url": "http://mirror.aarnet.edu.au/pub/TED-talks/911Mothers_2010W-480p.mp4",
    "task_status": "running"
  }
}
```

客户端不断查询是否分析结束

```
========== REQUEST ==========
GET /api/recognition/video/analyze/video_e2e_9c7e55cef480 HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

========== RESPONSE ==========
HTTP/1.1 200 OK
Content-Type: application/json

{
  "code": 0,
  "msg": "查询成功",
  "data": {
    "task_id": "video_task_20260807_003",
    "task_name": "video_e2e_9c7e55cef480",
    "source_url": "http://mirror.aarnet.edu.au/pub/TED-talks/911Mothers_2010W-480p.mp4",
    "task_status": "running",
    "create_time": "2026-08-07 08:16:07",
    "start_time": "2026-08-07 08:16:07",
    "end_time": "",
    "face_records": [],
    "error_info": ""
  }
}
.
.
.
========== REQUEST ==========
GET /api/recognition/video/analyze/video_e2e_9c7e55cef480 HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

========== RESPONSE ==========
HTTP/1.1 200 OK
Content-Type: application/json

{
  "code": 0,
  "msg": "查询成功",
  "data": {
    "task_id": "video_task_20260807_003",
    "task_name": "video_e2e_9c7e55cef480",
    "source_url": "http://mirror.aarnet.edu.au/pub/TED-talks/911Mothers_2010W-480p.mp4",
    "task_status": "finished",
    "create_time": "2026-08-07 08:16:07",
    "start_time": "2026-08-07 08:16:07",
    "end_time": "2026-08-07 08:16:33",
    "face_records": [],
    "error_info": ""
  }
}
```

客户端查询不存在的分析任务

```
========== REQUEST ==========
GET /api/recognition/video/analyze/not_existing_video_e2e_task HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

========== RESPONSE ==========
HTTP/1.1 404 Not Found
Content-Type: application/json

{
  "code": 1001,
  "msg": "Video analysis task not found",
  "data": {}
}
```