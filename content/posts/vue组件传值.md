---
title: "vue组件间传值需要注意的地方"
date: 2023-10-17
draft: false

tags: ["js"]
categories: ["font","vue"]
---



## 子组件接收父组件的函数

```js
const props = defineProps({
    scrollToBottom: {
        type: Function,
        required: false
    }
})
```

## 父组件传函数

```js
//消息框自动滚动到底部
function scrollToBottom() {
    setTimeout(() => {
        window.scrollTo({
            top: document.body.scrollHeight,
            left: 0,
            behavior: 'smooth'
        })
    }, 100)
}

<chatSearch :scrollToBottom="scrollToBottom"></chatSearch>
```

### 注意事项

- 无法立刻渲染问题我使用了一个延迟100，尝试过nextick，但是无效
-  behavior: 'smooth'，设置后确保window的滚动太丝滑滚动
- defineProps定义的函数必须声明一个props来接收后使用
- required: false，表示父组件中对于该项可传可不传

### 额外注意

>监听事件最好在组件卸载前主动清除，以免性能过分消耗