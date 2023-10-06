---
title: "两种快熟部署项目的方法"
date: 2023-09-28
draft: false

tags: ["docker"]
categories: ["other"]
---


## 两种快熟部署项目的方法

>- **CI/CD自动化构建镜像并部署**
>  这是最推荐的方式,可以防止手动操作带来的错误,实现自动化和标准化的部署流程。特别适合生产环境。
>
>- **DockerCompose 本地开发迭代**
>
>  在开发环境下,使用DockerCompose直接挂载宿主机目录到容器,实现代码热加载。这样可以省去反复重建镜像的时间,提高开发效率。

#### 前端Dockerfile

```dockerfile
FROM nginx:latest

# 移除默认的Nginx配置文件
RUN rm /etc/nginx/conf.d/default.conf

# 将前端项目的静态文件复制到 Nginx 的默认站点目录
COPY /home/qjw/blog/font/dist/ /usr/share/nginx/html/

# 复制自定义的Nginx配置文件到/etc/nginx/conf.d/目录下
COPY /home/qjw/blog/font/nginx.conf /etc/nginx/conf.d/

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### 后端Dockerfile

```dockerfile
# 使用官方 Python 镜像
FROM python:3.10.7

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY back/ /app

# 配置国内镜像
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

#更新pip
RUN pip install --upgrade pip

# 安装项目依赖项
RUN pip install -r requirements.txt

# 执行 Django 数据库迁移
RUN python manage.py migrate

# 启动 Django 项目
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

#### docker-compose.yml

```dockerfile
version: '3'
services:
  backend:
    build:
      context: .
      dockerfile: back/Dockerfile
    volumes:
      - /home/qjw/blog/back:/app  # 映射本地 Django 项目到容器的 /app 目录
    ports:
      - "8000:8000"  # 映射容器内部的 8000 端口到本地的 8000 端口
    network_mode: bridge
  frontend:
    build:
      context: .
      dockerfile: font/Dockerfile
    volumes:
      - /home/qjw/blog/font/dist:/usr/share/nginx/html
      - /home/qjw/blog/font/nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    network_mode: bridge                    
```

#### nginx.conf

```yaml
events {
    # 空的events块，可以在这里配置事件处理器，例如连接数，工作进程等
}

http {
    default_type application/octet-stream;
    sendfile on;
    keepalive_timeout 65;

    server {
        listen 80;
        server_name 8.134.109.184; # 替换成你的域名或服务器 IP 地址

        root /usr/share/nginx/html; # 替换成你的 Vue.js 项目的 dist 目录路径

        location / {
            index  index.html index.htm;
            try_files $uri $uri/ /index.html;
        }

        location ~* \.(js|css|png|jpg|jpeg|gif|ico|json)$ {
            expires max;
            add_header Cache-Control "public, max-age=31536000";
            types {
                application/javascript js;
                text/css css;
            }
        }
    }
}
```

#### 注意：

>docker-compose构建项目时会自动自动创建默认的网络,所以一般建议指定特定的网络