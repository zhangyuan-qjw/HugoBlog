---
title: "python中文件相对路径引发的问题"
date: 2023-11-06
draft: false

categories: ["python"]
---

## 需求如下：

>- 主项目目录是task_prompter，其中mian为次级文件夹
>- 项目根目录下的detectMessages.py文件下我需要调用main目录下的一个send.py文件方法
>- send方法包含一个相对路径的引用
>- 最终导致在detectMessages文件下的相对引用与send文件下的相对引用不一致出现了报错

#### 具体完善代码如下：

```python
current_dir = os.getcwd()
path = os.path.join(current_dir, "htmlTemplate", modelName, f"{modelName}.html")
with open(path, "r", encoding="utf-8") as html_file:
	html_content = html_file.read()
```

使用绝对路径解决，但是有点呆板！

`这种报错很难发现问题，一开始确实很莫名奇妙，需要谨慎使用相对路径的代码调用！`
