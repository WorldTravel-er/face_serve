# Docker

## 1. Linux系统安装

```bash
curl -fsSL https://get.docker.com -o install-docker.sh # 下载安装脚本
cat install-docker.sh # 查看安装脚本内容
sudo sh install-docker.sh # 安装Docker

# 验证安装
docker --version

```

## 2. Docker Hub

docker**官方镜像库：**https://hub.docker.com

网站镜像站：https://docker.fxxk.dedyn.io

```bash
# 从镜像站中拉取镜像
sudo docker pull docker.io/library/nginx:latest
#			registry(仓库地址)/namespace/镜像名:版本号
```

如果遇到网络问题，修改配置文件，设置镜像站，具体步骤如下：

1. 控制台输入命令

```bash
sudo vi /etc/docker/daemon.json # 打开配置文件
```

2. 将下面代码粘贴到文件中：

```json
{
	"registry-mirrors":[
		"https://docker.m.daocloud.io",
		"https://docker.1panel.live",
		"https://hub.rat.dev"
	]
}
```

3. 点击`ESC`，输入：`wq!回车`
4. 控制台输入命令，重启docker服务

```bash
sudo service docker restart
```

## 3. 常用命令

```bash
sudo docker images #列举所有下载过的Docker镜像
sudo docker rmi 镜像名/id # 删除镜像
sudo docker rm -f 容器名/id # 强制删除容器
sudo docker ps # 查看进程状况，正在运行的容器
sudo docker ps -a # 查看进程状况，所有容器，包括停止的
sudo docker volume list # 查询所有挂载卷
sudo docker volume rm 挂载卷 # 删除卷
sudo docker volume prune -a # 删除所有没有任何容器在使用的卷
sudo docker inspect 容器名/id # 打印容器的信息
sudo docker logs 容器名/id -f # 查看日志,-f表示滚动查看，并同步刷新
docker build -t hmch3n/face_api:latest . # 在当前文件夹构建镜像
```

## 4. 使用镜像并创建运行容器

```bash
sudo docker create -p 80:80 (镜像名/id) # 只创建不启动
sudo docker run nginx(镜像名/id) # 如果镜像不存在会自动拉取镜像
sudo docker -d run nginx(镜像名/id) # 容器后台执行，不阻塞当前窗口，否则控制台一直打印日志信息
sudo docker -p 80:81 run nginx(镜像名/id) # 将宿主机的80端口映射到容器的81端口
# 绑定挂载
sudo docker -v 宿主机目录:容器内目录 run nginx(镜像名/id) # 将宿主机目录和容器内目录绑定
# 命名卷挂载
sudo docker volume create 命名卷 # 创建存储空间（命名卷）
sudo docker volume inspect 挂载卷 # 显示挂载卷的位置
sudo docker -v 命名卷:容器内目录 run nginx(镜像名/id) # 将宿挂载卷和容器内目录绑定

sudo docker run (镜像名/id) -e XXX_XXX=xxx -e XXX_XXX=xxx # 传入其他参数作为环境变量
sudo docker run --name XXX nginx(镜像名/id) # 给容器设置唯一名字
sudo docker run -it --rm nginx(镜像名/id)  # -it:控制台进入容器进行交互 --rm:容器停止时删除掉
sudo docker run --restart always/unless-stopped (镜像名/id)# 容器停止了就立即重启/手动停止不重启
```

## 5. 容器启动与停止

```bash
sudo docker start (镜像名/id)
sudo docker stop (镜像名/id)
```

## 6. 查看容器内进程信息

```bash
docker exec 容器id linux命令
docker exec -it 容器id /bin/sh # 进入容器控制台
```

## 7. Dockerfile制作镜像

1. 创建Dockerfile文件，所有dockerfile第一行都是`FROM`，选择一个基础镜像（自己的镜像从哪个镜像的基础上构建而来），在docker hub 上选择python镜像

   ![image-20260806220852311](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260806220852311.png)

2. 从另外一个 Docker 镜像里面复制文件，镜像里安装uv

   <img src="C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260806221815839.png" alt="image-20260806221815839" style="zoom:80%;" />

3. uv以及其他设置：

   ```
   ENV UV_COMPILE_BYTECODE=1 # 安装完 Python 包以后，自动生成：.pyc字节码文件(__pycache__/)。
   ENV UV_LINK_MODE=copy # 这是 uv 安装包时的策略，稳定
   ENV PYTHONUNBUFFERED=1 # 降低输出延迟
   ENV PATH="/app/.venv/bin:$PATH" # 将虚拟环境加到PATH
   ```

   ![image-20260806222744011](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260806222744011.png)

4. `WORKDIR /app` ，可以将WORKDIR理解为cd，切换到工作目录

   ![image-20260806222804846](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260806222804846.png)

5. 安装系统依赖，并清理缓存

   ![image-20260806223158241](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260806223158241.png)

6. 将代码文件拷贝到工作目录中，

   ![image-20260806223430559](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260806223430559.png)

7. 安装项目依赖。 --frozen：严格按照 `uv.lock` 安装；--no-dev：不安装开发依赖(pytest...)

   ```
   RUN uv sync --frozen --no-dev
   ```

   ![image-20260806224158866](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260806224158866.png)

8. `EXPOSE 8000`：容器准备监听 8000 端口

   ![image-20260806224235028](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260806224235028.png)

9. Docker 容器启动时默认执行的命令

   ![image-20260806224422332](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260806224422332.png)

## 8. docker-compose.yml

管理一个或多个容器，可以理解成一个或多个Docker run命令

```yaml
services: # 定义要运行的服务列表,这里只有一个face-api服务
  face-api: # 服务名称，不是容器名
    build: # 构建镜像
      context: . # 构建上下文是当前目录
      dockerfile: Dockerfile # 指定构建文件,这里是默认
    image: face-api:latest # 指定构建后的镜像名称:标签
    container_name: face-api # 指定容器名字
    ports: # 端口映射
      - "8000:8000"
    volumes: # 把宿主机目录挂载到容器
      - ./data/face_api:/app/data/face_api
      - ./models/onnx:/app/models/onnx:ro
    restart: unless-stopped # 除非你手动停止，否则 Docker 会自动启动它。
    healthcheck: # Docker 会定期检查：服务是否真的正常。
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
      interval: 30s
      timeout: 5s
      retries: 3
```

使用当前目录 Dockerfile 构建镜像，创建名为 `face-api` 的容器，将宿主机的数据和 ONNX 模型挂载进去，映射 8000 端口，并配置自动重启和健康检查。

## 9. 启动

```
docker compose up -d --build # 无gpu环境启动
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build # gpu启动
docker compose logs -f face-api # 看日志
# 验证
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

## 10. 将镜像推送到docker hub上

1. 控制台输入：`docker login`，登录账号
2. 在当前文件夹构建镜像：`docker build -t hmch3n/face_api:latest .`

## 11. 推送失败

国内网络往往推送失败，换阿里云 ACR registry 。

在阿里云 ACR 控制台准备仓库：

```
容器镜像服务 ACR → 个人版实例 → 选择地域（华东1（杭州）） → 命名空间（hmCh3n） → 创建命名空间(face-api) → 选择私有/公开
```

操作指南：

![image-20260807110120771](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260807110120771.png)