---
title: "Java反射机制和注解"
date: 2023-11-18
draft: false

tags: ["java"]
---

## 注解

>进阶部分，读取注解

### 运行时创建对象？

当结合Java注解和反射机制时，你可以通过自定义注解来标记类，并使用反射机制在运行时创建对象。以下是一个简单的示例，假设你有一个自定义注解 `MyAnnotation`：

```java
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

// 定义自定义注解
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface MyAnnotation {
    String value() default ""; // 可以包含一些参数
}
```

然后，你可以使用这个注解来标记类：

```java
// 使用自定义注解
@MyAnnotation("Hello, I'm annotated!")
public class MyClass {
    public void displayInfo() {
        System.out.println("This is an annotated class.");
    }
}
```

接下来，你可以编写一个工具类，利用反射机制根据注解信息动态创建对象：

```java
import java.lang.reflect.Constructor;

public class AnnotationProcessor {
    public static Object createAnnotatedObject(Class<?> clazz) throws Exception {
        // 检查是否使用了MyAnnotation注解
        if (clazz.isAnnotationPresent(MyAnnotation.class)) {
            // 获取注解信息
            MyAnnotation annotation = clazz.getAnnotation(MyAnnotation.class);
            String value = annotation.value();

            // 使用默认构造方法创建对象
            Constructor<?> constructor = clazz.getDeclaredConstructor();
            return constructor.newInstance();
        } else {
            throw new IllegalArgumentException("Class is not annotated with MyAnnotation");
        }
    }

    public static void main(String[] args) {
        try {
            // 创建被注解的类的对象
            MyClass myObject = (MyClass) createAnnotatedObject(MyClass.class);
            myObject.displayInfo();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

在这个示例中，`AnnotationProcessor`类的`createAnnotatedObject`方法接受一个`Class`对象，并检查是否标记了`MyAnnotation`注解。如果标记了注解，就获取注解的信息，并使用默认构造方法创建对象。

在`main`方法中，我们调用了`createAnnotatedObject`方法来创建一个被注解类的对象，并调用其方法。

需要注意的是，这只是一个简单的示例，实际使用时可能需要更多的处理逻辑，特别是如果需要处理带参数的构造方法等情况。此外，需要考虑异常处理和错误情况。

### AOP切面编程

```java
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

// 目标接口
interface MyService {
    void doSomething();
}

// 目标实现类
class MyServiceImpl implements MyService {
    public void doSomething() {
        System.out.println("MyService - Doing something...");
    }
}

// 切面类
class MyAspect {
    public void beforeAdvice() {
        System.out.println("Before advice executed.");
    }
}

// 动态代理处理器
class MyProxyHandler implements InvocationHandler {
    private Object target;
    private MyAspect aspect;

    public MyProxyHandler(Object target, MyAspect aspect) {
        this.target = target;
        this.aspect = aspect;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        // 在方法调用前执行切面逻辑
        aspect.beforeAdvice();

        // 调用目标方法
        return method.invoke(target, args);
    }
}

public class MainApp {
    public static void main(String[] args) {
        // 创建目标实例和切面实例
        MyService myService = new MyServiceImpl();
        MyAspect myAspect = new MyAspect();

        // 创建动态代理
        MyService proxy = (MyService) Proxy.newProxyInstance(
                MainApp.class.getClassLoader(),
                new Class[]{MyService.class},
                new MyProxyHandler(myService, myAspect)
        );

        // 调用代理方法
        proxy.doSomething();
    }
}
```



## 反射

>Java的反射是指在运行时动态地获取类的信息、调用类的方法、操作类的属性等能力。反射机制允许程序在运行时通过一个类的名称来获取该类的相关信息，并且可以在运行时操作类的属性、方法和构造方法。这提供了一种灵活的方式，可以在编译时无法确定的情况下，动态地操作类的对象。

## Class对象

>在Java中，`Class`类是一个非常重要的类，它用于表示和封装Java中的类和接口的元数据信息。每个类（包括基本数据类型）在运行时都会有一个对应的`Class`对象，这个对象包含了该类的各种信息，如类的名称、字段、方法、构造方法等。

以下是一些`Class`类的常用方法和概念：

1. **获取Class对象：**
   - 通过类名获取：`Class.forName("com.example.MyClass")`
   - 通过对象的getClass方法：`obj.getClass()`
   - 通过类字面常量：`MyClass.class`
2. **获取类的信息：**
   - `getName()`：获取类的名称。
   - `getPackage()`：获取类所在的包。
   - `getModifiers()`：获取类的修饰符（public、private、static等）。
3. **获取类的成员信息：**
   - `getFields()`：获取类的public字段。
   - `getDeclaredFields()`：获取类的所有字段。
   - `getMethods()`：获取类的public方法。
   - `getDeclaredMethods()`：获取类的所有方法。
   - `getConstructors()`：获取类的public构造方法。
   - `getDeclaredConstructors()`：获取类的所有构造方法。
4. **创建实例：**
   - `newInstance()`：通过`Class`对象创建类的实例（已过时，推荐使用`Constructor`的`newInstance`方法）。
5. **其他方法：**
   - `isAssignableFrom(Class<?> cls)`：判断一个类是否可以赋值给另一个类。
   - `newInstance()`：创建类的实例。

##  Java代理

- 连接：https://xie.infoq.cn/article/9a9387805a496e1485dc8430f
