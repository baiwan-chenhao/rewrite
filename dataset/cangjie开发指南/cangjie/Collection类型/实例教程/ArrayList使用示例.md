ArrayList 的 append/insert 函数
ArrayList 中添加元素的方法如下：

import std.collection.*

main() {
    var list: ArrayList<Int64> = ArrayList<Int64>(10) //创建一个容量为 10 的 ArrayList
    var arr: Array<Int64> = [1, 2, 3]
    list.appendAll(arr) // list: [1, 2, 3]
    list.set(1, 120) // list: [1, 120, 3]
    var b = list.get(2)
    print("b=${b.getOrThrow()},")
    list.insert(1, 12) // list: [1, 12, 120, 3]
    var c = list.get(2)
    print("c=${c.getOrThrow()},")
    var arr1: Array<Int64> = [1,2,3]
    list.insertAll(1, arr1) // list: [1, 1, 2, 3, 12, 120, 3]
    var d = list.get(2)
    print("d=${d.getOrThrow()}")
    return 0
}
运行结果如下:

b=3,c=120,d=2

ArrayList 的 get/set 函数
此用例展示了如何使用 get 方法获取 ArrayList 中对应索引的值，以及使用 set 方法修改值。

代码如下：

import std.collection.*
main() {
    var list = ArrayList<Int64>([97, 100]) // list: [97, 100]
    list.set(1, 120) // list: [97, 120]
    var b = list.get(1)
    print("b=${b.getOrThrow()}")
    return 0
}
运行结果如下：


b=120


ArrayList 的 remove/clear/slice 函数
此用例展示了 ArrayList 的 remove/clear/slice 函数的使用方法。

代码如下：

import std.collection.*
main() {
    var list: ArrayList<Int64> = ArrayList<Int64>(97, 100, 99) // Function call syntactic sugar of variable-length
    list.remove(1) // list: [97, 99]
    var b = list.get(1)
    print("b=${b.getOrThrow()},")
    list.clear()
    list.append(11) // list: [97, 99, 11]
    var arr: Array<Int64> = [1, 2, 3]
    list.insertAll(0, arr) // list: [1, 2, 3, 97, 99]
    var g = list.get(0)
    print("g=${g.getOrThrow()},")
    let r: Range<Int64> = 1..=2 : 1
    var sublist: ArrayList<Int64> = list.slice(r) // sublist: [2, 3]
    var m = sublist.get(0)
    print("m=${m.getOrThrow()}")
    return 0
}
运行结果如下:


b=99,g=1,m=2