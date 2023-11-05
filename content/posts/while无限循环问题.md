---
title: "while无限循环问题"
date: 2023-11-05
draft: false

categories: ["python"]
---

### 问题：当我在程序中使用一个while(true)无限循环时，以便监听某个事件发生，其cpu性能会有不必要地消耗？我该如何使其阻塞并确保该事件发生时取消阻塞呢？

>解决方案有很多，常见的有线程睡眠和线程通信等待机制

## 以下使用线程通信解决方案

```python
import threading

class SharedResource:
    def __init__(self):
        self.condition = threading.Condition()  # 创建条件变量
        self.resource = None

    def set_resource(self, resource):
        with self.condition:
            self.resource = resource
            self.condition.notify()  # 唤醒等待的线程

    def get_resource(self):
        with self.condition:
            while self.resource is None:
                self.condition.wait()  # 等待条件变量满足
            return self.resource

def producer(shared_resource):
    resource = "example resource"
    shared_resource.set_resource(resource)

def consumer(shared_resource):
    resource = shared_resource.get_resource()
    print("Consumer got resource:", resource)

# 创建共享资源对象
shared_resource = SharedResource()

# 创建生产者线程和消费者线程
producer_thread = threading.Thread(target=producer, args=(shared_resource,))
consumer_thread = threading.Thread(target=consumer, args=(shared_resource,))

# 启动线程
producer_thread.start()
consumer_thread.start()

# 等待线程结束
producer_thread.join()
consumer_thread.join()
```

