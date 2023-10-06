---
title: "Passing value between vue components"
date: 2023-09-29
draft: false

tags: ["ts"]
categories: ["vue","front"]
---

### Parent component modifies the value of child component

>The properties in the props object are read-only properties and cannot be modifies within the component

#### Child component

```ts
const props = defineProps({
    newRole: {
        default: '邱',
        type: String,
    },
})
```

#### Parent component

```vue
x <chatSearch newRole=''></chatSearch>
```

### Child component modifies the value of Parent component 

>The properties in the props object are read-only properties and cannot be modifies within the component

```vue
//此处是父组件中引入的子组件
<ChildrenView  v-model:num="num"/>

//定义数据
let num=ref(10);//定义num为10,传递给子组
```

```html
<script setup>
    //子组件接收父组件传递过来的数据
    const props=defineProps({
        num:number;
    });
    console.log(props.num)//接收过来的数据num=10

    const emit=defineEmits(["update:num"]);//自定义的更新num事件
    const changeNum=()=>{
        emit("update:num",100);//触发自定义事件，将父组件的num修改为100
    }
</script>
```

