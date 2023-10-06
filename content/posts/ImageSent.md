---
title: "Use Django to store images sent from the front end"
date: 2023-09-24
draft: false

tags: ["python","ts"]
categories: ["chatGPT"]
---

## 使用django如何储存前端传来的图片？

在Django中，您可以使用以下步骤将前端传来的图片储存起来：

1. 在您的Django项目中创建一个模型来储存图片，例如：

```
arduinoCopy codefrom django.db import models

class Image(models.Model):
    name = models.CharField(max_length=255)
    image_file = models.ImageField(upload_to='images/')
```

这个模型有一个名称字段和一个ImageField字段，后者是Django特有的字段，用于储存图片。

1. 在您的视图中，您需要将上传的图片从请求中提取出来，并创建一个Image实例来储存它。以下是一个基本的示例视图：

```
arduinoCopy codefrom django.shortcuts import render
from .models import Image

def upload_image(request):
    if request.method == 'POST':
        image_file = request.FILES['image_file']
        name = request.POST['name']
        image = Image.objects.create(name=name, image_file=image_file)
        return render(request, 'upload_success.html', {'image': image})
    return render(request, 'upload_form.html')
```

这个视图假设您有一个名为"upload_form.html"的模板来渲染上传表单，以及一个名为"upload_success.html"的模板来渲染成功上传的页面。

1. 最后，在您的模板中，您需要确保您的表单使用"enctype='multipart/form-data'"来支持文件上传。以下是一个基本的示例模板：

```
phpCopy code<form action="{% url 'upload_image' %}" method="post" enctype="multipart/form-data">
    {% csrf_token %}
    <label for="name">Name:</label>
    <input type="text" name="name" id="name">
    <label for="image_file">Image:</label>
    <input type="file" name="image_file" id="image_file">
    <button type="submit">Upload</button>
</form>
```

这个模板包括一个文本输入框和一个文件输入框，以及一个提交按钮。注意"enctype='multipart/form-data'"这个属性是必须的来支持文件上传。

当您上传图片并提交表单后，Django将使用Image模型将图片存储在指定的目录中。在这个例子中，图片将被储存在"media/images/"目录中，因为我们在模型中使用了"upload_to='images/'"参数。

也就是说前端传来的图片存在后端创建的一个文件夹中，那数据库里会存图片的什么信息呢？

在Django中，当您使用ImageField来储存图片时，它实际上是将图片的路径（相对于媒体目录）存储在数据库中，而不是将整个图像文件存储在数据库中。这样做的好处是可以减少数据库的存储量，而且在获取图片时也可以更快地进行。

所以，当您在Django中使用ImageField来储存图片时，数据库中存储的是该图片的路径，而图片实际上是存在您设置的媒体目录中的。在您的例子中，图片将被储存在"media/images/"目录中，而在数据库中，Image模型的实例将储存图片的路径，例如"images/example.jpg"。

嗯，当前端请求图片时我只需要把响应的图片路径传过去就可以了，是吗？

是的，当您在前端请求图片时，您只需要将存储在数据库中的图片路径传递给模板或API视图，并将其插入到HTML或JSON中的相应标记或字段中即可。

例如，如果您的Image模型有一个名为image_file的ImageField字段，并且该字段储存了图片的路径，您可以在模板中使用以下方式来获取图片：

```
phpCopy code
<img src="{{ image.image_file.url }}" alt="{{ image.name }}">
```

在这个例子中，您可以使用"image.image_file.url"来获取该图片的URL路径，并将其插入到HTML的img标记中。

如果您是通过API来获取图片，您可以在您的API视图中返回一个JSON响应，并将图片的URL路径作为JSON对象的属性值，例如：

```
jsonCopy code{
    "name": "example",
    "image_file": "/media/images/example.jpg"
}
```

在这个例子中，您可以将"image_file"属性的值作为JSON对象返回给前端，并在前端使用它来显示或下载该图片。

嗯，好的。现在前端如何上传图片，使用axios上传。

使用axios上传图片到Django后端时，您可以使用以下步骤：

1. 在前端的HTML模板或组件中创建一个表单，包含一个文件输入框：

```
phpCopy code<form @submit.prevent="uploadImage">
  <input type="file" ref="image" accept="image/*">
  <button type="submit">Upload</button>
</form>
```

这个表单包含一个文件输入框和一个提交按钮。在这个例子中，我们使用了Vue.js中的@submit.prevent来阻止默认的表单提交行为，并将表单的提交事件绑定到一个名为uploadImage的方法上。

1. 在Vue.js组件中编写uploadImage方法，使用axios库将图片文件上传到Django后端：

```
javascriptCopy codeimport axios from 'axios';

export default {
  methods: {
    async uploadImage() {
      const image = this.$refs.image.files[0];
      const formData = new FormData();
      formData.append('image_file', image);
      try {
        const response = await axios.post('/upload_image/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        console.log(response.data);
      } catch (error) {
        console.log(error);
      }
    }
  }
}
```

这个方法将获取文件输入框中的文件，并使用FormData对象来创建一个包含文件的表单数据。然后，它将使用axios.post方法将表单数据上传到Django后端的/upload_image/ URL。

注意，当使用FormData对象上传文件时，必须将请求头的Content-Type设置为'multipart/form-data'，以便服务器知道该请求包含文件数据。

1. 在Django后端的视图中处理上传的图片文件：

```
pythonCopy codefrom django.http import JsonResponse

def upload_image(request):
    if request.method == 'POST':
        image_file = request.FILES['image_file']
        # 这里进行处理，例如将文件储存在指定目录中
        # 然后返回一个JSON响应，包含图片的路径等信息
        return JsonResponse({'status': 'ok', 'image_file': 'media/images/' + image_file.name})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
```

在这个视图中，它将使用request.FILES来获取上传的文件，并将其储存在您指定的目录中。然后，它将返回一个JSON响应，其中包含图片的路径等信息。

注意，如果您使用了Django的默认设置，上传的文件将被储存在'media/'目录中。因此，在这个例子中，我们将返回的图片路径设置为'media/images/' + 文件名。

这就是使用axios在Vue.js中上传图片到Django后端的基本步骤。当然，这只是一个示例，您可以根据您的需求进行更改和定制。