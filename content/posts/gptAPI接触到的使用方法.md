---
title: "gptAPI接触到的几种使用方法"
date: 2023-12-03
draft: false

tags: ["python"]
categories: ["chatGPT"]
---

## 简单对话聊天使用

>略

## 返回JSON数据格式

### 使用关键字

- response_format={ "type": "json_object" }

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ]
)
print(response.choices[0].message.content)
```

```json
"content": "{\"winner\": \"Los Angeles Dodgers\"}"`
```

### 提示API

提示 API 端点于 2023 年 7 月收到最终更新，并且具有与新的聊天完成端点不同的界面。输入不是消息列表，而是称为 的自由格式文本字符串`prompt`。

API 调用示例如下所示：

```python
from openai import OpenAI
client = OpenAI()

response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="Write a tagline for an ice cream shop.")
```

## 根据类型模型返回python实例

>可按需转换成JSONG格式
>
>**response_model=KnowledgeGraph**

```python
from models import KnowledgeGraph

completion: KnowledgeGraph = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    response_model=KnowledgeGraph,
)
# 转换成字典
response_data = completion.model_dump()
```

### 类型模型

```python
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    createdDate: str = Field(
        ..., description="知识图谱的创建日期"
    )
    lastUpdated: str = Field(
        ..., description="知识图谱最后更新的日期"
    )
    description: str = Field(..., description="知识图谱的描述")

class Node(BaseModel):
    id: str = Field(..., description="节点的唯一标识符")
    label: str = Field(..., description="节点的标签")
    type: str = Field(..., description="节点的类型")
    color: str = Field(..., description="节点的颜色")
    properties: Dict[str, Any] = Field(
        {}, description="节点的附加属性"
    )

class Edge(BaseModel):
    # 警告:请注意,这里字段名里使用的是"from_",而不是"from"
    from_: str = Field(..., alias="from", description="关系的起始节点ID")
    to: str = Field(..., description="关系的目标节点ID")
    relationship: str = Field(..., description="节点之间的关系类型")
    direction: str = Field(..., description="关系的方向")
    color: str = Field(..., description="关系的颜色")
    properties: Dict[str, Any] = Field(
        {}, description="关系的附加属性"
    )
class KnowledgeGraph(BaseModel):
    """
    生成一个包含实体和关系的知识图谱。
    使用颜色来帮助区分不同类型/分类的节点和边。
    总是提供浅淡的过渡色,与黑色字体配合效果良好。
    """
    metadata: Metadata = Field(..., description="知识图谱的元数据")
    nodes: List[Node] = Field(..., description="知识图谱中的Node列表")
    edges: List[Edge] = Field(..., description="知识图谱中的Edge列表")
```

