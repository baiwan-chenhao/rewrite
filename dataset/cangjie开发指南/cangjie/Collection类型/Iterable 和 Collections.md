Iterable 和 Collections
前面我们已经了解过 Range、Array、ArrayList，它们都可以使用 for-in 进行遍历操作，那么对一个用户自定义类型，能不能实现类似的遍历操作呢？答案是可以的。

Range、Array、ArrayList 其实都是通过 Iterable 来支持 for-in 语法的。

Iterable 是如下形式（只展示了核心代码）的一个内置 interface。

interface Iterable<T> {
    func iterator(): Iterator<T>
    ...
}
iterator 函数要求返回的 Iterator 类型是如下形式（只展示了核心代码）的另一个内置 interface。

interface Iterator<T> <: Iterable<T> {
    mut func next(): Option<T>
    ...
}
我们可以使用 for-in 语法来遍历任何一个实现了 Iterable 接口类型的实例。

假设有这样一个 for-in 代码。

let list = [1, 2, 3]
for (i in list) {
    println(i)
}
那么它等价于如下形式的 while 代码。

let list = [1, 2, 3]
var it = list.iterator()
while (true) {
    match (it.next()) {
        case Some(i) => println(i)
        case None => break
    }
}
另外一种常见的遍历 Iterable 类型的方法是使用 while-let，比如上面 while 代码的另一种等价写法是：

let list = [1, 2, 3]
var it = list.iterator()
while (let Some(i) <- it.next()) {
    println(i)
}
Array、ArrayList、HashSet、HashMap 类型都实现了 Iterable，因此我们都可以将其用在 for-in 或者 while-let 中。